#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone forward residual dynamics training for demo episodes."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Callable, cast

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
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

from isaaclab.utils.io import dump_yaml
from uwlab_rl.rsl_rl.forward_dynamics_cfg import ForwardDynamicsTrainerCfg
from uwlab_rl.rsl_rl.forward_dynamics_utils import (
    ForwardDynamicsResidualMLP,
    InverseDynamicsTransitionDataset,
    collate_inverse_dynamics,
)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg_dict(cfg_path: str | None) -> dict | None:
    if cfg_path is None:
        return None
    if cfg_path.endswith(".pt"):
        payload = torch.load(cfg_path, map_location="cpu")
        if isinstance(payload, dict) and "cfg" in payload:
            payload = payload["cfg"]
        if not isinstance(payload, dict):
            raise ValueError("Config checkpoint must contain a dict or a dict under 'cfg'.")
        return payload
    if cfg_path.endswith((".yaml", ".yml")):
        container = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        if not isinstance(container, dict):
            raise ValueError("YAML config must contain a dictionary at the top level.")
        return container
    raise ValueError("Unsupported config file type. Use .pt, .yaml, or .yml.")


def apply_cfg_overrides(
    cfg: ForwardDynamicsTrainerCfg,
    cfg_dict: dict | None,
    overrides: list[str],
) -> None:
    merged = OmegaConf.create(asdict(cfg))
    if cfg_dict:
        merged = OmegaConf.merge(merged, OmegaConf.create(cfg_dict))
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    merged_dict = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(merged_dict, dict):
        raise ValueError("Merged config must resolve to a dictionary.")
    from_dict_fn = cast(Callable[[dict], None], getattr(cfg, "from_dict"))
    from_dict_fn(merged_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward residual dynamics training.")
    parser.add_argument("--config", type=str, default=None, help="Path to a config .pt/.yaml file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args, overrides = parser.parse_known_args()

    cfg = ForwardDynamicsTrainerCfg()
    cfg_dict = load_cfg_dict(args.config)
    apply_cfg_overrides(cfg, cfg_dict, overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    train_episode_paths = cfg.data.train_episode_paths
    assert train_episode_paths is not None, "Training episode paths must be provided via config or CLI."
    train_dataset = InverseDynamicsTransitionDataset(train_episode_paths, cfg.data.obs_keys)
    if len(train_dataset) == 0:
        raise RuntimeError("No transitions found for inverse dynamics training.")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        drop_last=len(train_dataset) >= int(cfg.data.batch_size),
        collate_fn=collate_inverse_dynamics,
    )

    val_loader = None
    if cfg.data.validation_episode_paths:
        val_dataset = InverseDynamicsTransitionDataset(cfg.data.validation_episode_paths, cfg.data.obs_keys)
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.data.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                drop_last=False,
                collate_fn=collate_inverse_dynamics,
            )

    sample = train_dataset[0]
    state_dim = int(sample["state_t"].shape[-1])
    action_dim = int(sample["action_t"].shape[-1])
    model = ForwardDynamicsResidualMLP(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
    ).to(device)

    # Estimate state/action normalization from a few batches.
    if bool(getattr(cfg.model, "normalize_state_action", False)):
        num_batches = int(min(getattr(cfg.model, "norm_num_batches", 0), len(train_loader)))
        if num_batches <= 0:
            raise ValueError("norm_num_batches must be > 0 when normalize_state_action=True.")
        state_sum = torch.zeros(state_dim, device=device, dtype=torch.float64)
        state_sumsq = torch.zeros(state_dim, device=device, dtype=torch.float64)
        action_sum = torch.zeros(action_dim, device=device, dtype=torch.float64)
        action_sumsq = torch.zeros(action_dim, device=device, dtype=torch.float64)
        sample_count = 0.0
        with torch.no_grad():
            it = iter(train_loader)
            for _ in range(num_batches):
                batch = next(it)
                state_t = batch["state_t"].to(device=device, dtype=torch.float64)
                action_t = batch["action_t"].to(device=device, dtype=torch.float64)
                state_sum += state_t.sum(dim=0)
                state_sumsq += (state_t * state_t).sum(dim=0)
                action_sum += action_t.sum(dim=0)
                action_sumsq += (action_t * action_t).sum(dim=0)
                sample_count += float(state_t.shape[0])
        denom = max(sample_count, 1.0)
        state_mean = (state_sum / denom).to(dtype=torch.float32)
        action_mean = (action_sum / denom).to(dtype=torch.float32)
        state_var = state_sumsq / denom - (state_sum / denom) * (state_sum / denom)
        action_var = action_sumsq / denom - (action_sum / denom) * (action_sum / denom)
        min_std = float(getattr(cfg.model, "norm_min_std", 1.0e-6))
        state_std = torch.sqrt(state_var.clamp_min(min_std * min_std)).to(dtype=torch.float32)
        action_std = torch.sqrt(action_var.clamp_min(min_std * min_std)).to(dtype=torch.float32)
    else:
        state_mean = torch.zeros(state_dim, device=device, dtype=torch.float32)
        state_std = torch.ones(state_dim, device=device, dtype=torch.float32)
        action_mean = torch.zeros(action_dim, device=device, dtype=torch.float32)
        action_std = torch.ones(action_dim, device=device, dtype=torch.float32)
    model.set_input_normalization(state_mean, state_std, action_mean, action_std)
    optimizer_class = getattr(torch.optim, cfg.optim.optimizer_class)
    optimizer = optimizer_class(
        model.parameters(),
        lr=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.optim.use_amp and torch.cuda.is_available())
    loss_fn = nn.MSELoss()

    log_root = os.path.join("logs", "rsl_rl", cfg.logging.experiment_name)
    os.makedirs(log_root, exist_ok=True)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.logging.run_name and cfg.logging.run_name != "":
        run_name = cfg.logging.run_name
    log_dir = os.path.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    params_dir = os.path.join(log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    dump_yaml(os.path.join(params_dir, "trainer.yaml"), asdict(cfg))

    wandb_run = None
    if cfg.logging.use_wandb:
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

    def compute_validation_loss() -> float:
        if val_loader is None:
            return float("nan")
        model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for batch in val_loader:
                state_t = batch["state_t"].to(device)
                state_tp1 = batch["state_tp1"].to(device)
                action_t = batch["action_t"].to(device)
                target_residual = state_tp1 - state_t
                with torch.cuda.amp.autocast(enabled=cfg.optim.use_amp and torch.cuda.is_available()):
                    pred_residual = model(state_t=state_t, action_t=action_t)
                    loss = loss_fn(pred_residual, target_residual)
                batch_size = int(action_t.shape[0])
                total_loss += float(loss.item()) * batch_size
                total_count += batch_size
        model.train()
        if total_count == 0:
            return float("nan")
        return total_loss / float(total_count)

    total_steps = 0
    total_steps_target = int(cfg.optim.num_steps)
    progress_bar = tqdm(total=total_steps_target, desc="Forward dynamics updates", unit="step")
    for epoch in range(int(math.ceil(cfg.optim.num_steps / max(len(train_loader), 1)))):
        for batch in train_loader:
            state_t = batch["state_t"].to(device)
            state_tp1 = batch["state_tp1"].to(device)
            action_t = batch["action_t"].to(device)
            target_residual = state_tp1 - state_t

            with torch.cuda.amp.autocast(enabled=cfg.optim.use_amp and torch.cuda.is_available()):
                pred_residual = model(state_t=state_t, action_t=action_t)
                loss = loss_fn(pred_residual, target_residual)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            if total_steps % cfg.logging.log_interval == 0:
                if wandb_run is not None:
                    lr_value = optimizer.param_groups[0]["lr"] if optimizer.param_groups else None
                    wandb_run.log(
                        {
                            "train/residual_loss": float(loss.item()),
                            "train/grad_norm": float(grad_norm),
                            "train/lr": lr_value,
                            "train/step": total_steps,
                        }
                    )
            if val_loader is not None and cfg.logging.val_interval > 0:
                if total_steps % cfg.logging.val_interval == 0:
                    val_loss = compute_validation_loss()
                    if wandb_run is not None:
                        wandb_run.log({"val/residual_loss": val_loss, "val/step": total_steps})
            if total_steps % cfg.logging.save_interval == 0:
                ckpt_path = os.path.join(log_dir, f"model_{total_steps:06d}.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "trainer_cfg": asdict(cfg),
                        "normalization": model.get_input_normalization_payload(),
                        "meta": {
                            "state_dim": state_dim,
                            "action_dim": action_dim,
                        },
                    },
                    ckpt_path,
                )

            total_steps += 1
            progress_bar.update(1)
            if total_steps >= total_steps_target:
                break
        if total_steps >= total_steps_target:
            break
        if epoch > total_steps_target:
            break
    progress_bar.close()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
