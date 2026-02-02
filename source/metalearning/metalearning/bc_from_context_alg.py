from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from rsl_rl.algorithms.ppo import PPO
from uwlab_rl.rsl_rl.long_context_ac import LongContextActorCritic
from uwlab_rl.rsl_rl.lr_utils import cosine_annealing_with_warmup
from uwlab_rl.rsl_rl.metaleanring_cfg import BCFromContextWarmStartCfg


class BCFromContext:
    """Behavior cloning warm start using environment context episodes."""

    def __init__(self, ppo: PPO, cfg: BCFromContextWarmStartCfg) -> None:
        self.ppo = ppo
        self.cfg = cfg
        self._ddp_policy: DistributedDataParallel | None = None
        raw_policy = ppo.policy
        if isinstance(raw_policy, DistributedDataParallel):
            self._ddp_policy = raw_policy
            raw_policy = raw_policy.module
        self.policy = cast(LongContextActorCritic, raw_policy)
        if not isinstance(self.policy, LongContextActorCritic):
            raise TypeError("BCFromContext expects LongContextActorCritic policy.")
        self.device = next(self.policy.parameters()).device
        self._categorical_actions = self.policy.action_distribution == "categorical"
        self.use_amp = bool(cfg.use_amp and torch.cuda.is_available())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.optimizer = self._build_optimizer()
        self._base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self._lr_scheduler = self._build_lr_scheduler()
        self.loss_fn = nn.MSELoss()
        self.obs_keys: list[str] | None = None
        self._wandb_metrics_defined = False
        self._is_multi_gpu, self._gpu_world_size, self._gpu_global_rank = self._resolve_distributed_state()
        self._reduce_parameters = getattr(self.ppo, "reduce_parameters", None)

    def _build_optimizer(self) -> optim.Optimizer:
        optimizer_class = getattr(optim, self.cfg.optimizer_class)
        return optimizer_class(
            self.policy.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
            eps=self.cfg.eps,
        )

    def _build_lr_scheduler(self) -> optim.lr_scheduler.LambdaLR | None:
        warmup_steps = int(self.cfg.lr_warmup_steps)
        total_steps = int(self.cfg.num_steps)
        if warmup_steps <= 0 or total_steps <= 0:
            return None
        return cosine_annealing_with_warmup(self.optimizer, warmup_steps, total_steps)

    def warm_start(self, env: Any) -> dict[str, float]:
        context = _get_env_context(env)
        episodes = context.episodes
        self._validate_episode_counts(episodes)
        if self._ddp_policy is not None:
            self._ddp_policy.train()
        else:
            self.policy.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        num_updates = 0
        for step_idx in range(self.cfg.num_steps):
            batch = self._sample_episode_batch(episodes, self.cfg.num_episodes_per_batch)
            data = self._prepare_batch(batch)
            loss_value, grad_norm = self._train_on_batch(**data)
            self._apply_lr_schedule(step_idx)
            if self._should_log() and step_idx % 10 == 0:
                print(f"Loss: {loss_value}, Grad Norm: {grad_norm}")
            loss_value, grad_norm = self._sync_metrics(loss_value, grad_norm)
            total_loss += loss_value
            total_grad_norm += grad_norm
            num_updates += 1
            if self._should_log():
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
                if self._categorical_actions:
                    logits = self._forward_from_context(
                        batch_demo_obs,
                        batch_demo_actions,
                        batch_demo_rewards,
                        batch_demo_lengths,
                        batch_current_obs,
                        return_logits=True,
                    )
                    loss = self._categorical_loss(logits, batch_target)
                else:
                    pred = self._forward_from_context(
                        batch_demo_obs,
                        batch_demo_actions,
                        batch_demo_rewards,
                        batch_demo_lengths,
                        batch_current_obs,
                        return_logits=False,
                    )
                    loss = self.loss_fn(pred, batch_target)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self._reduce_gradients_if_needed()
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            batch_loss += loss.item()
            batch_grad_norm += float(grad_norm)
            num_minibatches += 1
        denom = max(num_minibatches, 1)
        return batch_loss / denom, batch_grad_norm / denom

    def _predict_action_logits(
        self,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_rewards: torch.Tensor,
        demo_lengths: torch.Tensor,
        current_obs: torch.Tensor,
    ) -> torch.Tensor:
        if self.policy._use_transformer_actor():
            tokens, padding_mask, token_indices = self.policy._build_transformer_tokens_from_context(
                demo_obs,
                demo_actions,
                demo_rewards,
                demo_lengths,
                current_obs,
            )
            obs_tensor = self.policy._normalize_transformer_tokens(tokens)
            return self.policy.actor(
                obs_tensor,
                padding_mask=padding_mask,
                token_indices=token_indices,
            )
        if self.policy.context_encoder is None:
            raise RuntimeError("Context encoder is not initialized.")
        context_hidden = self.policy.context_encoder(demo_obs, demo_actions, demo_rewards)
        # Advanced indexing to pick the last valid context token per sample.
        context_hidden = context_hidden[
            torch.arange(demo_obs.shape[0], device=demo_obs.device),
            demo_lengths.squeeze(-1).to(dtype=torch.long) - 1,
            :,
        ]
        obs_tensor = self.policy._merge_obs_with_context(current_obs, context_hidden)
        obs_tensor = self.policy.actor_obs_normalizer(obs_tensor)
        return self.policy.actor(obs_tensor)

    def _forward_from_context(
        self,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_rewards: torch.Tensor,
        demo_lengths: torch.Tensor,
        current_obs: torch.Tensor,
        return_logits: bool,
    ) -> torch.Tensor:
        policy = self._ddp_policy if self._ddp_policy is not None else self.policy
        return policy(
            demo_obs,
            demo_actions,
            demo_rewards,
            demo_lengths,
            current_obs,
            return_logits=return_logits,
        )

    def _categorical_loss(self, logits: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        action_bins = self.policy.action_bins
        if action_bins is None:
            raise RuntimeError("Categorical actions require action_bins.")
        action_indices = self._actions_to_indices(target_actions, action_bins)
        if logits.ndim == 2:
            splits = torch.split(logits, action_bins, dim=-1)
        elif logits.ndim == 3:
            splits = [logits[:, idx, :bins] for idx, bins in enumerate(action_bins)]
        else:
            raise ValueError(f"Expected logits to be 2D or 3D, got {tuple(logits.shape)}")
        losses = []
        for idx, per_dim_logits in enumerate(splits):
            losses.append(F.cross_entropy(per_dim_logits, action_indices[:, idx], reduction="mean"))
        return torch.stack(losses).mean()

    def _actions_to_indices(
        self,
        actions: torch.Tensor,
        action_bins: tuple[int, ...],
    ) -> torch.Tensor:
        if actions.dtype == torch.long:
            idx = actions
        else:
            if self.policy.action_bin_values is not None:
                idxs = []
                for dim, values in enumerate(self.policy.action_bin_values):
                    values = values.to(device=actions.device, dtype=actions.dtype).view(1, -1)
                    target = actions[..., dim].unsqueeze(-1)
                    idxs.append((target - values).abs().argmin(dim=-1))
                idx = torch.stack(idxs, dim=-1)
            else:
                idx = actions.round().to(torch.long)
        idx_clamped = [idx[..., dim].clamp(0, bins - 1) for dim, bins in enumerate(action_bins)]
        return torch.stack(idx_clamped, dim=-1)

    def _resolve_minibatch_size(self, num_samples: int) -> int:
        if self.cfg.minibatch_size is not None:
            return max(1, int(self.cfg.minibatch_size))
        return max(1, int(num_samples / max(self.cfg.num_minibatches, 1)))

    def _apply_lr_schedule(self, step_idx: int) -> None:
        if self._lr_scheduler is None:
            return
        self._lr_scheduler.step(step_idx + 1)

    def _resolve_distributed_state(self) -> tuple[bool, int, int]:
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not is_distributed:
            return False, 1, 0
        world_size = int(getattr(self.ppo, "gpu_world_size", torch.distributed.get_world_size()))
        global_rank = int(getattr(self.ppo, "gpu_global_rank", torch.distributed.get_rank()))
        return world_size > 1, world_size, global_rank

    def _reduce_gradients_if_needed(self) -> None:
        if not self._is_multi_gpu or self._ddp_policy is not None:
            return
        if callable(self._reduce_parameters):
            self._reduce_parameters()
            return
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        for param in self.policy.parameters():
            if param.grad is None:
                continue
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
            param.grad.div_(self._gpu_world_size)

    def _sync_metrics(self, loss_value: float, grad_norm: float) -> tuple[float, float]:
        if not self._is_multi_gpu:
            return loss_value, grad_norm
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return loss_value, grad_norm
        metrics = torch.tensor([loss_value, grad_norm], device=self.device, dtype=torch.float32)
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
        metrics = metrics / float(self._gpu_world_size)
        return float(metrics[0].item()), float(metrics[1].item())

    def _validate_episode_counts(self, episodes: list[dict[str, Any]]) -> None:
        if not self._is_multi_gpu:
            if not episodes:
                raise ValueError("No demo episodes found in env.context.")
            return
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            if not episodes:
                raise ValueError("No demo episodes found in env.context.")
            return
        local_count = torch.tensor([len(episodes)], device=self.device, dtype=torch.int64)
        counts = [torch.zeros_like(local_count) for _ in range(self._gpu_world_size)]
        torch.distributed.all_gather(counts, local_count)
        count_values = [int(count.item()) for count in counts]
        if any(count == 0 for count in count_values):
            raise ValueError(f"BC warm-start requires demo episodes on all ranks. counts={count_values}")

    def _should_log(self) -> bool:
        return not self._is_multi_gpu or self._gpu_global_rank == 0

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
