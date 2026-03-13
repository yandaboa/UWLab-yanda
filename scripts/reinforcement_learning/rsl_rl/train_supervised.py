#!/usr/bin/env python3
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a supervised MLP policy from state-action dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

try:
    from uwlab_rl.rsl_rl.supervised_mlp_policy import SupervisedMLPPolicy
except ModuleNotFoundError:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _source_root = os.path.join(_repo_root, "source", "uwlab_rl")
    if _source_root not in sys.path:
        sys.path.append(_source_root)
    from uwlab_rl.rsl_rl.supervised_mlp_policy import SupervisedMLPPolicy


class StateActionDataset(Dataset):
    """Simple tensor-backed dataset for supervised state-action learning."""

    def __init__(self, states: torch.Tensor, actions: torch.Tensor):
        if states.ndim != 2:
            raise ValueError(f"Expected states tensor with shape [N, state_dim], got {tuple(states.shape)}")
        if actions.ndim != 2:
            raise ValueError(f"Expected actions tensor with shape [N, action_dim], got {tuple(actions.shape)}")
        if states.shape[0] != actions.shape[0]:
            raise ValueError(
                f"States and actions must have same number of samples, got {states.shape[0]} and {actions.shape[0]}"
            )

        self.states = states.float()
        self.actions = actions.float()

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.states[index], self.actions[index]


@dataclass
class TrainConfig:
    dataset_path: str
    output_dir: str
    hidden_dims: list[int]
    loss_type: str
    normalize_action_targets: bool
    action_norm_min_std: float
    log_std_min: float
    log_std_max: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    val_split: float
    num_workers: int
    seed: int
    save_every_epochs: int
    save_last: bool
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str | None


def load_config(config_path: str) -> TrainConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    dataset_cfg = raw.get("dataset", {})
    model_cfg = raw.get("model", {})
    train_cfg = raw.get("training", {})

    return TrainConfig(
        dataset_path=str(dataset_cfg["path"]),
        output_dir=str(train_cfg.get("output_dir", "logs/rsl_rl/supervised")),
        hidden_dims=list(model_cfg.get("hidden_dims", [512, 256, 128, 64])),
        loss_type=str(model_cfg.get("loss_type", "gaussian_nll")),
        normalize_action_targets=bool(model_cfg.get("normalize_action_targets", True)),
        action_norm_min_std=float(model_cfg.get("action_norm_min_std", 1.0e-6)),
        log_std_min=float(model_cfg.get("log_std_min", -5.0)),
        log_std_max=float(model_cfg.get("log_std_max", 2.0)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        batch_size=int(train_cfg.get("batch_size", 2048)),
        epochs=int(train_cfg.get("epochs", 50)),
        val_split=float(train_cfg.get("val_split", 0.1)),
        num_workers=int(train_cfg.get("num_workers", 4)),
        seed=int(train_cfg.get("seed", 42)),
        save_every_epochs=int(train_cfg.get("save_every_epochs", 1)),
        save_last=bool(train_cfg.get("save_last", True)),
        use_wandb=bool(train_cfg.get("use_wandb", True)),
        wandb_project=str(train_cfg.get("wandb_project", "supervised_gaussian_policy")),
        wandb_run_name=train_cfg.get("wandb_run_name"),
    )


def make_dataloaders(
    states: torch.Tensor,
    actions: torch.Tensor,
    val_split: float,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    dataset = StateActionDataset(states=states, actions=actions)
    num_samples = len(dataset)
    if num_samples < 2:
        raise ValueError("Dataset needs at least 2 samples for train/validation split.")

    val_size = max(1, int(num_samples * val_split))
    train_size = num_samples - val_size
    if train_size <= 0:
        raise ValueError(
            f"Validation split leaves no training samples: num_samples={num_samples}, val_split={val_split}"
        )

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def evaluate(model: SupervisedMLPPolicy, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for states, actions in loader:
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            loss = model.compute_loss(states, actions)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite validation loss detected: {loss.item()}")

            batch_size = states.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    return total_loss / max(1, total_count)


def save_checkpoint(
    path: str,
    model: SupervisedMLPPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    state_dim: int,
    action_dim: int,
    cfg: TrainConfig,
    best_val_loss: float,
    config_path: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dims": cfg.hidden_dims,
            "loss_type": cfg.loss_type,
            "log_std_min": cfg.log_std_min,
            "log_std_max": cfg.log_std_max,
            "best_val_loss": best_val_loss,
            "action_norm_mean": model.action_norm_mean.detach().cpu(),
            "action_norm_std": model.action_norm_std.detach().cpu(),
            "normalize_action_targets": cfg.normalize_action_targets,
            "config": asdict(cfg),
            "config_path": os.path.abspath(config_path),
            "dataset_path": os.path.abspath(cfg.dataset_path),
        },
        path,
    )


def init_wandb(
    cfg: TrainConfig,
    config_path: str,
    run_dir: str,
    num_samples: int,
    state_dim: int,
    action_dim: int,
) -> tuple[Any, Any]:
    """Initialize wandb for supervised training when requested."""
    if not cfg.use_wandb:
        return None, None

    try:
        import wandb  # type: ignore
    except ImportError:
        print("[INFO] Wandb requested but not installed; skipping wandb logging.")
        return None, None

    run_name = cfg.wandb_run_name or os.path.basename(run_dir)
    run = wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        dir=run_dir,
        config={
            "dataset_path": os.path.abspath(cfg.dataset_path),
            "config_path": os.path.abspath(config_path),
            "num_samples": num_samples,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "model": {
                "hidden_dims": cfg.hidden_dims,
                "loss_type": cfg.loss_type,
                "normalize_action_targets": cfg.normalize_action_targets,
                "action_norm_min_std": cfg.action_norm_min_std,
                "log_std_min": cfg.log_std_min,
                "log_std_max": cfg.log_std_max,
            },
            "training": {
                "output_dir": os.path.abspath(cfg.output_dir),
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "val_split": cfg.val_split,
                "num_workers": cfg.num_workers,
                "seed": cfg.seed,
                "save_every_epochs": cfg.save_every_epochs,
                "save_last": cfg.save_last,
            },
        },
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("checkpoint/*", step_metric="epoch")
    return run, wandb


def log_checkpoint_to_wandb(
    wandb_run: Any,
    wandb_module: Any,
    checkpoint_path: str,
    checkpoint_kind: str,
    epoch: int,
    val_loss: float,
) -> None:
    """Log checkpoint metadata and sync the saved file to wandb."""
    if wandb_run is None or wandb_module is None:
        return

    abs_path = os.path.abspath(checkpoint_path)
    wandb_module.log(
        {
            "epoch": epoch,
            f"checkpoint/{checkpoint_kind}_val_loss": val_loss,
        }
    )
    wandb_module.save(abs_path, base_path=os.path.dirname(abs_path), policy="now")


def main():
    parser = argparse.ArgumentParser(description="Train a supervised MLP policy from collected dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "train_supervised.yaml"),
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if cfg.save_every_epochs <= 0:
        raise ValueError(f"save_every_epochs must be > 0, got {cfg.save_every_epochs}")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    dataset = torch.load(cfg.dataset_path, map_location="cpu")
    if "states" not in dataset or "actions" not in dataset:
        raise KeyError(f"Dataset at {cfg.dataset_path} must contain 'states' and 'actions' tensors.")

    states: torch.Tensor = dataset["states"].float()
    actions: torch.Tensor = dataset["actions"].float()

    if states.ndim != 2:
        raise ValueError(f"Expected dataset['states'] to have shape [N, state_dim], got {tuple(states.shape)}")
    if actions.ndim != 2:
        raise ValueError(f"Expected dataset['actions'] to have shape [N, action_dim], got {tuple(actions.shape)}")
    if states.shape[0] != actions.shape[0]:
        raise ValueError(
            f"Dataset states/actions sample mismatch: {states.shape[0]} vs {actions.shape[0]}"
        )

    train_loader, val_loader = make_dataloaders(
        states=states,
        actions=actions,
        val_split=cfg.val_split,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    state_dim = int(states.shape[1])
    action_dim = int(actions.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model : SupervisedMLPPolicy = SupervisedMLPPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=cfg.hidden_dims,
        loss_type=cfg.loss_type,
        log_std_min=cfg.log_std_min,
        log_std_max=cfg.log_std_max,
    ).to(device)
    if cfg.normalize_action_targets:
        action_norm_mean = actions.mean(dim=0)
        action_norm_std = actions.std(dim=0, unbiased=False).clamp_min(cfg.action_norm_min_std)
    else:
        action_norm_mean = torch.zeros(action_dim, dtype=torch.float32)
        action_norm_std = torch.ones(action_dim, dtype=torch.float32)
    model.set_action_normalization(action_norm_mean, action_norm_std)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(cfg.output_dir, f"supervised_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    wandb_run, wandb_module = init_wandb(
        cfg=cfg,
        config_path=args.config,
        run_dir=run_dir,
        num_samples=int(states.shape[0]),
        state_dim=state_dim,
        action_dim=action_dim,
    )

    best_val_loss = float("inf")
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    epoch_iterator = tqdm(range(cfg.epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_iterator:
        model.train()
        total_train_loss = 0.0
        total_train_count = 0

        for batch_states, batch_actions in train_loader:
            batch_states = batch_states.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)

            loss = model.compute_loss(batch_states, batch_actions)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss detected at epoch={epoch + 1}, "
                    f"loss={loss.item()}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = batch_states.shape[0]
            total_train_loss += float(loss.item()) * batch_size
            total_train_count += batch_size

        train_loss = total_train_loss / max(1, total_train_count)
        val_loss = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        epoch_iterator.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
        print(f"[Epoch {epoch + 1:03d}/{cfg.epochs:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if wandb_run is not None and wandb_module is not None:
            log_payload: dict[str, float] = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/lr": float(optimizer.param_groups[0]["lr"]),
            }
            if model.log_std_param is not None:
                log_payload["train/log_std_mean"] = float(model.log_std_param.detach().mean().item())
                log_payload["train/log_std_min"] = float(model.log_std_param.detach().min().item())
                log_payload["train/log_std_max"] = float(model.log_std_param.detach().max().item())
            wandb_module.log(log_payload)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(run_dir, "best_model.pt")
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                state_dim=state_dim,
                action_dim=action_dim,
                cfg=cfg,
                best_val_loss=best_val_loss,
                config_path=args.config,
            )
            log_checkpoint_to_wandb(
                wandb_run=wandb_run,
                wandb_module=wandb_module,
                checkpoint_path=best_checkpoint_path,
                checkpoint_kind="best_model",
                epoch=epoch + 1,
                val_loss=best_val_loss,
            )

        if (epoch + 1) % cfg.save_every_epochs == 0:
            epoch_checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch + 1:03d}.pt")
            save_checkpoint(
                path=epoch_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                state_dim=state_dim,
                action_dim=action_dim,
                cfg=cfg,
                best_val_loss=best_val_loss,
                config_path=args.config,
            )
            log_checkpoint_to_wandb(
                wandb_run=wandb_run,
                wandb_module=wandb_module,
                checkpoint_path=epoch_checkpoint_path,
                checkpoint_kind="epoch_model",
                epoch=epoch + 1,
                val_loss=val_loss,
            )

    if cfg.save_last:
        last_checkpoint_path = os.path.join(run_dir, "last_model.pt")
        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=cfg.epochs,
            state_dim=state_dim,
            action_dim=action_dim,
            cfg=cfg,
            best_val_loss=best_val_loss,
            config_path=args.config,
        )
        log_checkpoint_to_wandb(
            wandb_run=wandb_run,
            wandb_module=wandb_module,
            checkpoint_path=last_checkpoint_path,
            checkpoint_kind="last_model",
            epoch=cfg.epochs,
            val_loss=history["val_loss"][-1],
        )

    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(run_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val_loss
        wandb_run.summary["output_dir"] = os.path.abspath(run_dir)
        wandb_run.finish()

    print(f"[INFO] Training complete. Best validation loss: {best_val_loss:.6f}")
    print(f"[INFO] Saved outputs to: {os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main()