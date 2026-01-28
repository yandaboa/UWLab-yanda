from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from uwlab_rl.rsl_rl.long_context_ac import LongContextActorCritic
from uwlab_rl.rsl_rl.metaleanring_cfg import BCFromContextWarmStartCfg


class BCFromContext:
    """Behavior cloning warm start using environment context episodes."""

    def __init__(self, ppo: PPO, cfg: BCFromContextWarmStartCfg) -> None:
        self.ppo = ppo
        self.cfg = cfg
        self.policy = cast(LongContextActorCritic, ppo.policy)
        if not isinstance(self.policy, LongContextActorCritic):
            raise TypeError("BCFromContext expects LongContextActorCritic policy.")
        self.device = next(self.policy.parameters()).device
        self.use_amp = bool(cfg.use_amp and torch.cuda.is_available())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.optimizer = self._build_optimizer()
        self._base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.loss_fn = nn.MSELoss()
        self.obs_keys: list[str] | None = None
        self._wandb_metrics_defined = False

    def _build_optimizer(self) -> optim.Optimizer:
        optimizer_class = getattr(optim, self.cfg.optimizer_class)
        return optimizer_class(
            self.policy.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
            eps=self.cfg.eps,
        )

    def warm_start(self, env: Any) -> dict[str, float]:
        context = _get_env_context(env)
        episodes = context.episodes
        if not episodes:
            raise ValueError("No demo episodes found in env.context.")
        self.policy.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        num_updates = 0
        for step_idx in range(self.cfg.num_steps):
            self._apply_lr_warmup(step_idx)
            batch = self._sample_episode_batch(episodes, self.cfg.num_episodes_per_batch)
            data = self._prepare_batch(batch)
            loss_value, grad_norm = self._train_on_batch(**data)
            total_loss += loss_value
            total_grad_norm += grad_norm
            num_updates += 1
            self._log_wandb(step_idx, loss_value, grad_norm)
        denom = max(num_updates, 1)
        return {
            "bc_loss": total_loss / denom,
            "bc_grad_norm": total_grad_norm / denom,
        }

    def _sample_episode_batch(self, episodes: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        if count <= 0:
            raise ValueError("num_episodes_per_batch must be positive.")
        if len(episodes) >= count:
            indices = torch.randperm(len(episodes))[:count].tolist()
        else:
            indices = torch.randint(0, len(episodes), (count,)).tolist()
        return [episodes[idx] for idx in indices]

    def _prepare_batch(self, episodes: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        demo_obs_list = []
        demo_actions_list = []
        demo_rewards_list = []
        lengths = []
        for episode in episodes:
            obs = episode.get("obs")
            actions = episode.get("actions")
            if not isinstance(actions, torch.Tensor):
                raise TypeError("Episode actions must be a tensor.")
            rewards = episode.get("rewards")
            length = int(episode.get("length", actions.shape[0]))
            lengths.append(length)
            obs_seq = _flatten_obs(obs, self.obs_keys)
            if self.obs_keys is None and isinstance(obs, dict):
                self.obs_keys = list(obs.keys())
            demo_obs_list.append(obs_seq[:length])
            demo_actions_list.append(actions[:length].reshape(length, -1))
            demo_rewards_list.append(_coerce_rewards(rewards, length))
        max_len = max(lengths)
        demo_obs = _pad_sequence_list(demo_obs_list, max_len, self.device)
        demo_actions = _pad_sequence_list(demo_actions_list, max_len, self.device)
        demo_rewards = _pad_sequence_list(demo_rewards_list, max_len, self.device)
        demo_lengths = torch.tensor(lengths, device=self.device, dtype=torch.long).unsqueeze(-1)
        sample_episode_idx, sample_time_idx = _build_sample_indices(lengths, self.device)
        # Advanced indexing to pick per-sample current obs and actions.
        current_obs = demo_obs[sample_episode_idx, sample_time_idx]
        target_actions = demo_actions[sample_episode_idx, sample_time_idx]
        return {
            "demo_obs": demo_obs,
            "demo_actions": demo_actions,
            "demo_rewards": demo_rewards,
            "demo_lengths": demo_lengths,
            "sample_episode_idx": sample_episode_idx,
            "sample_time_idx": sample_time_idx,
            "current_obs": current_obs,
            "target_actions": target_actions,
        }

    def _train_on_batch(
        self,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_rewards: torch.Tensor,
        demo_lengths: torch.Tensor,
        sample_episode_idx: torch.Tensor,
        sample_time_idx: torch.Tensor,
        current_obs: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> tuple[float, float]:
        num_samples = int(sample_episode_idx.shape[0])
        minibatch_size = self._resolve_minibatch_size(num_samples)
        permutation = torch.randperm(num_samples, device=self.device)
        batch_loss = 0.0
        batch_grad_norm = 0.0
        num_minibatches = 0
        for start in range(0, num_samples, minibatch_size):
            idx = permutation[start : start + minibatch_size]
            episode_idx = sample_episode_idx[idx]
            time_idx = sample_time_idx[idx]
            batch_demo_obs = demo_obs[episode_idx]
            batch_demo_actions = demo_actions[episode_idx]
            batch_demo_rewards = demo_rewards[episode_idx]
            batch_demo_lengths = demo_lengths[episode_idx]
            batch_current_obs = current_obs[idx]
            batch_target = target_actions[idx]
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred = self.policy.act_from_context(
                    batch_demo_obs,
                    batch_demo_actions,
                    batch_demo_rewards,
                    batch_demo_lengths,
                    batch_current_obs,
                )
                loss = self.loss_fn(pred, batch_target)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            batch_loss += loss.item()
            batch_grad_norm += float(grad_norm)
            num_minibatches += 1
        denom = max(num_minibatches, 1)
        return batch_loss / denom, batch_grad_norm / denom

    def _resolve_minibatch_size(self, num_samples: int) -> int:
        if self.cfg.minibatch_size is not None:
            return max(1, int(self.cfg.minibatch_size))
        return max(1, int(num_samples / max(self.cfg.num_minibatches, 1)))

    def _apply_lr_warmup(self, step_idx: int) -> None:
        warmup_steps = int(self.cfg.lr_warmup_steps)
        if warmup_steps <= 0:
            return
        progress = min(step_idx + 1, warmup_steps) / float(warmup_steps)
        start_ratio = float(self.cfg.lr_warmup_start_ratio)
        scale = start_ratio + (1.0 - start_ratio) * progress
        for base_lr, group in zip(self._base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def _log_wandb(self, step_idx: int, loss_value: float, grad_norm: float) -> None:
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        if wandb.run is None:
            return
        if not self._wandb_metrics_defined:
            wandb.define_metric("bc_warmstart/step")
            wandb.define_metric("bc_warmstart/*", step_metric="bc_warmstart/step")
            self._wandb_metrics_defined = True
        lr = None
        if self.optimizer.param_groups:
            lr = self.optimizer.param_groups[0].get("lr")
        wandb.log(
            {
                "bc_warmstart/loss": loss_value,
                "bc_warmstart/grad_norm": grad_norm,
                "bc_warmstart/lr": lr,
                "bc_warmstart/step": step_idx,
            }
        )


def _get_env_context(env: Any) -> Any:
    context = getattr(env, "context", None)
    if context is None:
        unwrapped = getattr(env, "unwrapped", None)
        context = getattr(unwrapped, "context", None) if unwrapped is not None else None
    if context is None:
        raise AttributeError("Env has no demo context. Expected env.context to be initialized.")
    return context


def _flatten_obs(obs: Any, obs_keys: list[str] | None) -> torch.Tensor:
    if isinstance(obs, dict):
        keys = obs_keys or list(obs.keys())
        flat_terms = []
        for key in keys:
            value = obs[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Observation key '{key}' is not a tensor.")
            flat_terms.append(value.reshape(value.shape[0], -1))
        return torch.cat(flat_terms, dim=-1)
    if isinstance(obs, torch.Tensor):
        return obs.reshape(obs.shape[0], -1)
    raise TypeError(f"Unsupported obs type: {type(obs)}")


def _coerce_rewards(rewards: Any, length: int) -> torch.Tensor:
    if rewards is None:
        return torch.zeros((length, 1))
    if isinstance(rewards, torch.Tensor):
        if rewards.ndim == 1:
            return rewards[:length].unsqueeze(-1)
        return rewards[:length].reshape(length, -1)
    raise TypeError("Episode rewards must be a tensor or None.")


def _pad_sequence_list(sequences: list[torch.Tensor], max_len: int, device: torch.device) -> torch.Tensor:
    if not sequences:
        raise ValueError("Cannot pad empty sequence list.")
    padded = []
    for seq in sequences:
        seq = seq.to(device)
        if seq.shape[0] < max_len:
            pad = torch.zeros((max_len - seq.shape[0], *seq.shape[1:]), device=device, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=0)
        padded.append(seq)
    return torch.stack(padded, dim=0)


def _build_sample_indices(lengths: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    episode_indices = []
    time_indices = []
    for idx, length in enumerate(lengths):
        episode_indices.extend([idx] * length)
        time_indices.extend(range(length))
    return (
        torch.tensor(episode_indices, device=device, dtype=torch.long),
        torch.tensor(time_indices, device=device, dtype=torch.long),
    )
