#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone supervised training for context-conditioned transformers."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime

import torch
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
from uwlab_rl.rsl_rl.lr_utils import build_lr_scheduler
from uwlab_rl.rsl_rl.supervised_context_cfg import SupervisedContextTrainerCfg
# from uwlab_tasks.manager_based.manipulation.from_demo.config.ur5e_robotiq_2f85.agents.supervised_context_cfg import SupervisedContextRunnerCfg
from uwlab_rl.rsl_rl.supervised_context_utils import (
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

from isaaclab.utils.io import dump_yaml


def _should_log(is_multi_gpu: bool, rank: int) -> bool:
    return not is_multi_gpu or rank == 0




def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised context training.")
    parser.add_argument("--config", type=str, default=None, help="Path to a config .pt/.yaml file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args, overrides = parser.parse_known_args()

    cfg = SupervisedContextTrainerCfg()
    cfg_dict = load_cfg_dict(args.config)
    apply_cfg_overrides(cfg, cfg_dict, overrides)
    if cfg.distributed.distributed and dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    is_multi_gpu, world_size, rank = resolve_distributed(cfg.distributed.distributed)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(args.seed + rank)

    train_episode_paths = cfg.data.train_episode_paths or cfg.data.episode_paths
    assert train_episode_paths is not None, "Training episode paths must be provided via config or CLI."
    validation_episode_paths = cfg.data.validation_episode_paths
    assert validation_episode_paths is not None, "Validation episode paths must be provided via config or CLI."
    dataset = ContextStepDataset(train_episode_paths, cfg.data.obs_keys)
    sampler = DistributedSampler(dataset) if is_multi_gpu else None
    effective_drop_last = len(dataset) >= int(cfg.data.batch_size)
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=cfg.data.shuffle if sampler is None else False, num_workers=cfg.data.num_workers, sampler=sampler, drop_last=effective_drop_last, collate_fn=collate_context_steps)

    if len(dataset) == 0:
        raise RuntimeError("No episodes found for supervised training.")
    val_loader = None 
    val_dataset = ContextStepDataset(validation_episode_paths, cfg.data.obs_keys)
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
        dump_yaml(os.path.join(params_dir, "trainer.yaml"), asdict(cfg))
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
                config=asdict(cfg),
            )

    total_steps = 0
    total_steps_target = int(cfg.optim.num_steps)
    progress_bar = None
    if _should_log(is_multi_gpu, rank):
        progress_bar = tqdm(total=total_steps_target, desc="Supervised updates", unit="step")
    def _compute_validation_loss() -> float:
        if val_loader is None:
            return float("nan")
        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        total_count = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for batch in val_loader:
                demo_obs = batch["demo_obs"].to(device)
                demo_actions = batch["demo_actions"].to(device)
                demo_rewards = batch["demo_rewards"].to(device)
                demo_lengths = batch["demo_lengths"].to(device)
                current_obs = batch["current_obs"].to(device)
                target_action = batch["target_action"].to(device)
                model_module = model.module if isinstance(model, DistributedDataParallel) else model
                with torch.cuda.amp.autocast(enabled=cfg.optim.use_amp and torch.cuda.is_available()):
                    if cfg.input.include_current_trajectory:
                        raise ValueError("include_current_trajectory=True is not supported yet.")
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
    for epoch in range(int(math.ceil(cfg.optim.num_steps / max(len(loader), 1)))):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            demo_obs = batch["demo_obs"].to(device)
            demo_actions = batch["demo_actions"].to(device)
            demo_rewards = batch["demo_rewards"].to(device)
            demo_lengths = batch["demo_lengths"].to(device)
            current_obs = batch["current_obs"].to(device)
            target_action = batch["target_action"].to(device)
            model_module = model.module if isinstance(model, DistributedDataParallel) else model
            model_module.train()
            with torch.cuda.amp.autocast(enabled=cfg.optim.use_amp and torch.cuda.is_available()):
                if cfg.input.include_current_trajectory:
                    raise ValueError("include_current_trajectory=True is not supported yet.")
                loss = model_module.compute_supervised_loss(
                    demo_obs=demo_obs,
                    demo_actions=demo_actions,
                    demo_rewards=demo_rewards,
                    demo_lengths=demo_lengths,
                    current_obs=current_obs,
                    target_action=target_action,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if total_steps % cfg.logging.log_interval == 0:
                reduced_loss = reduce_loss_if_needed(loss, is_multi_gpu)
                if _should_log(is_multi_gpu, rank):
                    grad_norm_value = float(grad_norm)
                    lr_value = optimizer.param_groups[0]["lr"] if optimizer.param_groups else None
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
                                "train/step": total_steps,
                            }
                        )
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
                payload["trainer_cfg"] = asdict(cfg)
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
