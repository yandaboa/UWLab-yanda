#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone supervised training for context-conditioned transformers."""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from datetime import datetime

import torch

torch.backends.cuda.enable_flash_sdp = True
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)  # prefer flash/mem-efficient

import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SOURCE_DIR = os.path.join(ROOT_DIR, "source")
PKG_DIRS = [
    SOURCE_DIR,
    os.path.join(SOURCE_DIR, "uwlab_rl"),
    os.path.join(SOURCE_DIR, "uwlab_tasks"),
]
for path in (ROOT_DIR, *PKG_DIRS):
    if path not in sys.path:
        sys.path.append(path)

from uwlab_rl.rsl_rl.context_sequence_policy import ContextSequencePolicy
from uwlab_rl.rsl_rl.io_utils import class_to_dict, dump_yaml
from uwlab_rl.rsl_rl.lr_utils import build_lr_scheduler
from uwlab_rl.rsl_rl.supervised_context_cfg import SupervisedContextTrainerCfg
# from uwlab_tasks.manager_based.manipulation.from_demo.config.ur5e_robotiq_2f85.agents.supervised_context_cfg import SupervisedContextRunnerCfg
from uwlab_rl.rsl_rl.supervised_context_utils import (
    EpisodeGroupDataset,
    SplineQueryEpisodeGroupDataset,
    ContextStepDataset,
    collate_context_steps,
    resolve_action_bin_values,
    load_action_discretization_spec,
    reduce_loss_if_needed,
    resolve_distributed,
    resolve_action_bins,
    seed_everything,
)
from scripts.reinforcement_learning.rsl_rl.supervised_context_cli_utils import (
    apply_cfg_overrides,
    load_cfg_dict,
)


def _should_log(is_multi_gpu: bool, rank: int) -> bool:
    return not is_multi_gpu or rank == 0


def _expand_episode_paths(paths: list[str] | None) -> list[str]:
    if paths is None:
        return []
    expanded: list[str] = []
    for item in paths:
        matches = sorted(glob.glob(item))
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(item)
    # De-duplicate while preserving order.
    unique_paths: list[str] = []
    seen = set()
    for path in expanded:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def _paths_use_episode_groups(paths: list[str]) -> bool:
    if len(paths) == 0:
        raise ValueError("Episode path list is empty.")
    grouped: bool | None = None
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        has_groups = isinstance(payload, dict) and "episode_groups" in payload
        has_episodes = isinstance(payload, dict) and "episodes" in payload
        if not has_groups and not has_episodes:
            raise ValueError(f"{path} must contain either 'episodes' or 'episode_groups'.")
        current_grouped = has_groups
        if grouped is None:
            grouped = current_grouped
        elif grouped != current_grouped:
            raise ValueError("Mixed dataset payloads are unsupported; use all grouped or all ungrouped files.")
    assert grouped is not None
    return grouped


def _paths_have_group_splines(paths: list[str]) -> bool:
    if len(paths) == 0:
        return False
    has_splines = True
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        has_group_splines = (
            isinstance(payload, dict)
            and "episode_groups" in payload
            and isinstance(payload.get("group_splines"), list)
        )
        has_splines = has_splines and has_group_splines
    return has_splines




def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised context training.")
    parser.add_argument("--config", type=str, default=None, help="Path to a config .pt/.yaml file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args, overrides = parser.parse_known_args()

    cfg = SupervisedContextTrainerCfg()
    cfg_dict = load_cfg_dict(args.config)
    apply_cfg_overrides(cfg, cfg_dict, overrides)
    trainer_cfg_dict = class_to_dict(cfg)
    if cfg.distributed.distributed and dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    is_multi_gpu, world_size, rank = resolve_distributed(cfg.distributed.distributed)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(args.seed + rank)

    train_episode_paths = _expand_episode_paths(cfg.data.train_episode_paths or cfg.data.episode_paths)
    validation_episode_paths = _expand_episode_paths(cfg.data.validation_episode_paths)
    if len(train_episode_paths) == 0:
        raise ValueError("No training episode paths configured.")
    if len(validation_episode_paths) == 0:
        raise ValueError("No validation episode paths configured.")
    if cfg.data.train_on_data_augs and cfg.input.include_current_trajectory:
        raise ValueError(
            "train_on_data_augs=True is not supported with include_current_trajectory=True "
            "for ContextStepDataset training."
        )
    use_episode_groups = _paths_use_episode_groups(train_episode_paths)
    print(f"INFO: Training on episode groups: {use_episode_groups}")
    dataset: ContextStepDataset | EpisodeGroupDataset | SplineQueryEpisodeGroupDataset
    train_group_dataset: EpisodeGroupDataset | SplineQueryEpisodeGroupDataset | None = None
    if use_episode_groups:
        assert not cfg.data.train_on_data_augs, (
            "train_on_data_augs=True is only supported for non-grouped ('episodes') datasets."
        )
        use_spline_queries = _paths_have_group_splines(train_episode_paths)
        if use_spline_queries:
            print("INFO: Using spline/reference query states/actions for grouped training.")
            train_group_dataset = SplineQueryEpisodeGroupDataset(
                train_episode_paths,
                cfg.data.obs_keys,
                randomize_context=True,
                base_seed=args.seed + rank,
                num_context_episodes=cfg.data.num_context_episodes,
                minimum_num_context_trajs=cfg.data.minimum_num_context_trajs,
                include_current_trajectory=cfg.input.include_current_trajectory,
            )
        else:
            train_group_dataset = EpisodeGroupDataset(
                train_episode_paths,
                cfg.data.obs_keys,
                randomize_context=True,
                base_seed=args.seed + rank,
                num_context_episodes=cfg.data.num_context_episodes,
                minimum_num_context_trajs=cfg.data.minimum_num_context_trajs,
                include_current_trajectory=cfg.input.include_current_trajectory,
            )
        dataset = train_group_dataset
    else:
        dataset = ContextStepDataset(
            train_episode_paths,
            cfg.data.obs_keys,
            include_data_augs=cfg.data.train_on_data_augs,
            include_current_trajectory=cfg.input.include_current_trajectory,
        )
    sampler = DistributedSampler(dataset) if is_multi_gpu else None
    if cfg.data.train_on_data_augs and sampler is None:
        assert cfg.data.shuffle, "train_on_data_augs=True requires data.shuffle=True for random mixing."
    effective_drop_last = len(dataset) >= int(cfg.data.batch_size)
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle if sampler is None else False, num_workers=cfg.data.num_workers, sampler=sampler, drop_last=effective_drop_last, collate_fn=collate_context_steps)

    if len(dataset) == 0:
        raise RuntimeError("No episodes found for supervised training.")
    val_loader = None
    use_episode_groups_val = _paths_use_episode_groups(validation_episode_paths)
    if use_episode_groups != use_episode_groups_val:
        raise ValueError("Training and validation datasets must both be grouped or both be ungrouped.")
    val_dataset: ContextStepDataset | EpisodeGroupDataset | SplineQueryEpisodeGroupDataset
    if use_episode_groups:
        use_spline_queries_val = _paths_have_group_splines(validation_episode_paths)
        if use_spline_queries_val:
            val_group_dataset = SplineQueryEpisodeGroupDataset(
                validation_episode_paths,
                cfg.data.obs_keys,
                randomize_context=False,
                base_seed=args.seed + 7919,
                num_context_episodes=cfg.data.num_context_episodes,
                minimum_num_context_trajs=cfg.data.minimum_num_context_trajs,
                include_current_trajectory=cfg.input.include_current_trajectory,
            )
        else:
            val_group_dataset = EpisodeGroupDataset(
                validation_episode_paths,
                cfg.data.obs_keys,
                randomize_context=False,
                base_seed=args.seed + 7919,
                num_context_episodes=cfg.data.num_context_episodes,
                minimum_num_context_trajs=cfg.data.minimum_num_context_trajs,
                include_current_trajectory=cfg.input.include_current_trajectory,
            )
        val_dataset = val_group_dataset
        assert train_group_dataset is not None
        observed_max_context_length = max(
            train_group_dataset.max_context_length, val_group_dataset.max_context_length
        )
        if cfg.data.max_context_length is None:
            cfg.data.max_context_length = observed_max_context_length
        else:
            cfg.data.max_context_length = min(int(cfg.data.max_context_length), observed_max_context_length)
        print(f"[INFO] Using grouped contexts with max_context_length={cfg.data.max_context_length}")
    else:
        val_dataset = ContextStepDataset(
            validation_episode_paths,
            cfg.data.obs_keys,
            include_current_trajectory=cfg.input.include_current_trajectory,
        )
    assert len(val_dataset) > 0, "No episodes found for supervised validation."
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_multi_gpu else None
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers, sampler=val_sampler, drop_last=False, collate_fn=collate_context_steps)
    sample = dataset[0]
    obs_dim = int(sample["obs"].shape[-1])
    action_dim = int(sample["actions"].shape[-1])
    reward_dim = int(sample["rewards"].shape[-1])
    cfg.model.num_actions = action_dim
    action_bins = None
    action_bin_values = None
    if cfg.model.action_distribution == "categorical":
        spec = load_action_discretization_spec(
            train_episode_paths,
            cfg.model.action_discretization_spec_path,
        )
        assert spec is not None, "Categorical actions require action discretization spec."
        action_bins = resolve_action_bins(spec, action_dim)
        action_bin_values = resolve_action_bin_values(spec, action_dim)

    # Estimate action normalization from a few training batches (optional).
    action_norm_mean = torch.zeros(action_dim, device=device, dtype=torch.float64)
    action_norm_sumsq = torch.zeros(action_dim, device=device, dtype=torch.float64)
    action_norm_count = torch.tensor(0.0, device=device, dtype=torch.float64)
    if (
        bool(getattr(cfg.model, "normalize_action_targets", False))
        and cfg.model.action_distribution in {"normal", "scalar"}
    ):
        num_batches = int(min(getattr(cfg.model, "action_norm_num_batches", 0), len(loader)))
        if num_batches <= 0:
            raise ValueError("action_norm_num_batches must be > 0 when normalize_action_targets=True.")
        with torch.no_grad():
            it = iter(loader)
            for _ in range(num_batches):
                batch = next(it)
                actions = batch["target_action"].to(device=device, dtype=torch.float64)
                action_norm_mean += actions.sum(dim=0)
                action_norm_sumsq += (actions * actions).sum(dim=0)
                action_norm_count += float(actions.shape[0])
        if is_multi_gpu:
            dist.all_reduce(action_norm_mean, op=dist.ReduceOp.SUM)
            dist.all_reduce(action_norm_sumsq, op=dist.ReduceOp.SUM)
            dist.all_reduce(action_norm_count, op=dist.ReduceOp.SUM)
        denom = action_norm_count.clamp_min(1.0)
        action_norm_mean = action_norm_mean / denom
        action_var = action_norm_sumsq / denom - action_norm_mean * action_norm_mean
        min_std = float(getattr(cfg.model, "action_norm_min_std", 1.0e-6))
        action_norm_std = torch.sqrt(action_var.clamp_min(min_std * min_std))
        action_norm_mean = action_norm_mean.to(dtype=torch.float32)
        action_norm_std = action_norm_std.to(dtype=torch.float32)
    else:
        action_norm_mean = torch.zeros(action_dim, device=device, dtype=torch.float32)
        action_norm_std = torch.ones(action_dim, device=device, dtype=torch.float32)
    print(f"Action normalization mean: {action_norm_mean}")
    print(f"Action normalization std: {action_norm_std}")
    model = ContextSequencePolicy(
        cfg,
        obs_dim,
        action_dim,
        reward_dim,
        action_bins=action_bins,
        action_bin_values=action_bin_values,
    ).to(device)
    if is_multi_gpu:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    model_module = model.module if isinstance(model, DistributedDataParallel) else model
    model_module.set_action_normalization(action_norm_mean, action_norm_std)
    optimizer_class = getattr(torch.optim, cfg.optim.optimizer_class)
    optimizer = optimizer_class(
        model.parameters(),
        lr=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
    )
    lr_scheduler = build_lr_scheduler(cfg.optim, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.optim.use_amp and torch.cuda.is_available())

    log_root = os.path.join("logs", "rsl_rl", cfg.logging.experiment_name)
    os.makedirs(log_root, exist_ok=True)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.logging.run_name and cfg.logging.run_name != "":
        run_name = f"{cfg.logging.run_name}"
    log_dir = os.path.join(log_root, run_name)
    if _should_log(is_multi_gpu, rank):
        os.makedirs(log_dir, exist_ok=True)
        params_dir = os.path.join(log_dir, "params")
        os.makedirs(params_dir, exist_ok=True)
        dump_yaml(os.path.join(params_dir, "trainer.yaml"), trainer_cfg_dict)
    wandb_run = None
    if _should_log(is_multi_gpu, rank) and cfg.logging.use_wandb:
        try:
            import wandb  # type: ignore
        except ImportError:
            wandb = None
        if wandb is not None:
            project_name = cfg.logging.log_project_name or cfg.logging.experiment_name
            wandb_run = wandb.init(
                project=project_name,
                name=cfg.logging.run_name or run_name,
                config=trainer_cfg_dict,
            )

    total_steps = 0
    total_steps_target = int(cfg.optim.num_steps)
    grad_accumulation_steps = int(getattr(cfg.optim, "grad_accumulation_steps", 1))
    assert grad_accumulation_steps >= 1, "optim.grad_accumulation_steps must be >= 1."
    progress_bar = None
    if _should_log(is_multi_gpu, rank):
        progress_bar = tqdm(total=total_steps_target, desc="Supervised updates", unit="step")
    max_val_steps = cfg.logging.num_validation_steps
    def _compute_validation_loss() -> float:
        if val_loader is None:
            return float("nan")
        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        total_count = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for val_step_idx, batch in enumerate(val_loader):
                if max_val_steps is not None and val_step_idx >= max_val_steps:
                    break
                demo_obs = batch["demo_obs"].to(device)
                demo_actions = batch["demo_actions"].to(device)
                demo_rewards = batch["demo_rewards"].to(device)
                demo_lengths = batch["demo_lengths"].to(device)
                current_obs = batch["current_obs"].to(device)
                target_action = batch["target_action"].to(device)
                model_module = model.module if isinstance(model, DistributedDataParallel) else model
                with torch.cuda.amp.autocast(enabled=cfg.optim.use_amp and torch.cuda.is_available()):
                    loss = model_module.compute_supervised_loss(
                        demo_obs=demo_obs,
                        demo_actions=demo_actions,
                        demo_rewards=demo_rewards,
                        demo_lengths=demo_lengths,
                        current_obs=current_obs,
                        target_action=target_action,
                    )
                batch_size = target_action.shape[0]
                total_loss += loss.detach() * batch_size
                total_count += batch_size
        if is_multi_gpu:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
        model.train()
        if total_count.item() == 0:
            return float("nan")
        return (total_loss / total_count).item()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(math.ceil((cfg.optim.num_steps * grad_accumulation_steps) / max(len(loader), 1)))):
        if sampler is not None:
            sampler.set_epoch(epoch)
        accum_batches = 0
        accum_loss_for_log = torch.tensor(0.0, device=device)
        accum_seq_len_sum = 0.0
        accum_seq_len_count = 0
        accum_seq_len_max = 0.0
        for batch_idx, batch in enumerate(loader):
            demo_obs = batch["demo_obs"].to(device)
            demo_actions = batch["demo_actions"].to(device)
            demo_rewards = batch["demo_rewards"].to(device)
            demo_lengths = batch["demo_lengths"].to(device)
            current_obs = batch["current_obs"].to(device)
            target_action = batch["target_action"].to(device)
            model_module = model.module if isinstance(model, DistributedDataParallel) else model
            model_module.train()
            with torch.cuda.amp.autocast(enabled=cfg.optim.use_amp and torch.cuda.is_available()):
                loss = model_module.compute_supervised_loss(
                    demo_obs=demo_obs,
                    demo_actions=demo_actions,
                    demo_rewards=demo_rewards,
                    demo_lengths=demo_lengths,
                    current_obs=current_obs,
                    target_action=target_action,
                )
            loss_for_backward = loss / grad_accumulation_steps

            scaler.scale(loss_for_backward).backward()
            accum_batches += 1
            accum_loss_for_log += loss.detach()
            accum_seq_len_sum += float(demo_lengths.float().sum().item())
            accum_seq_len_count += int(demo_lengths.numel())
            accum_seq_len_max = max(accum_seq_len_max, float(demo_lengths.float().max().item()))

            should_step = accum_batches >= grad_accumulation_steps or (batch_idx + 1) == len(loader)
            if not should_step:
                continue

            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if lr_scheduler is not None:
                lr_scheduler.step()

            if total_steps % cfg.logging.log_interval == 0:
                reduced_loss = reduce_loss_if_needed(accum_loss_for_log / accum_batches, is_multi_gpu)
                if _should_log(is_multi_gpu, rank):
                    grad_norm_value = float(grad_norm)
                    lr_value = optimizer.param_groups[0]["lr"] if optimizer.param_groups else None
                    seq_len_mean = accum_seq_len_sum / max(accum_seq_len_count, 1)
                    seq_len_max = accum_seq_len_max
                    # print(
                    #     f"[step {total_steps}] loss={reduced_loss.item():.6f} "
                    #     f"grad_norm={grad_norm_value:.6f} lr={lr_value}"
                    # )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": reduced_loss.item(),
                                "train/grad_norm": grad_norm_value,
                                "train/lr": lr_value,
                                "train/sequence_length_mean": seq_len_mean,
                                "train/sequence_length_max": seq_len_max,
                                "train/step": total_steps,
                            }
                        )
            accum_batches = 0
            accum_loss_for_log.zero_()
            accum_seq_len_sum = 0.0
            accum_seq_len_count = 0
            accum_seq_len_max = 0.0
            if val_loader is not None and cfg.logging.val_interval > 0:
                if total_steps % cfg.logging.val_interval == 0:
                    val_loss = _compute_validation_loss()
                    if _should_log(is_multi_gpu, rank):
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "val/loss": val_loss,
                                    "val/step": total_steps,
                                }
                            )
            if _should_log(is_multi_gpu, rank) and total_steps % cfg.logging.save_interval == 0:
                ckpt_path = os.path.join(log_dir, f"model_{total_steps:06d}.pt")
                model_module = model.module if isinstance(model, DistributedDataParallel) else model
                payload = model_module.get_state_dict_payload()
                payload["trainer_cfg"] = trainer_cfg_dict
                payload["meta"] = {
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "reward_dim": reward_dim,
                    "action_bins": action_bins,
                    "action_bin_values": None
                    if action_bin_values is None
                    else [values.tolist() for values in action_bin_values],
                }
                torch.save(payload, ckpt_path)
            total_steps += 1
            if progress_bar is not None:
                progress_bar.update(1)
            if total_steps >= cfg.optim.num_steps:
                break
        if total_steps >= cfg.optim.num_steps:
            break
    if progress_bar is not None:
        progress_bar.close()


if __name__ == "__main__":
    main()
