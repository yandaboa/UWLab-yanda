from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from tensordict import TensorDict
from uwlab_rl.rsl_rl.lr_utils import cosine_annealing_with_warmup, linear_warmup

class CompositeOptimizer:
    """Proxy optimizer that forwards to policy and encoder optimizers."""

    def __init__(self, policy_optimizer: optim.Optimizer, encoder_optimizer: optim.Optimizer) -> None:
        self.policy_optimizer = policy_optimizer
        self.encoder_optimizer = encoder_optimizer

    @property
    def policy_param_groups(self) -> list[dict[str, Any]]:
        return self.policy_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.policy_optimizer.zero_grad(set_to_none=set_to_none)
        self.encoder_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable | None = None) -> tuple[Any, Any]:
        policy_result = self.policy_optimizer.step(closure=closure)
        encoder_result = self.encoder_optimizer.step(closure=closure)
        return policy_result, encoder_result

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy_optimizer.state_dict(),
            "encoder": self.encoder_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "policy" in state_dict and "encoder" in state_dict:
            self.policy_optimizer.load_state_dict(state_dict["policy"])
            self.encoder_optimizer.load_state_dict(state_dict["encoder"])
        else:
            self.policy_optimizer.load_state_dict(state_dict)


class PPOWithLongContext(PPO):
    """PPO variant that uses a dedicated transformer optimizer."""

    def __init__(
        self,
        policy,
        transformer_optimizer_cfg: Any | None = None,
        num_learning_iterations: int | None = None,
        use_bf16_amp: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy, **kwargs)
        self.num_learning_iterations = num_learning_iterations

        # ---- BF16 AMP toggle ----
        # Enable only on CUDA; BF16 autocast is CUDA-only in torch.cuda.amp
        self.use_bf16_amp = bool(use_bf16_amp and torch.cuda.is_available())
        self.amp_dtype = torch.bfloat16

        cfg = transformer_optimizer_cfg or getattr(policy, "transformer_optimizer_cfg", None)
        context_encoder = getattr(policy, "context_encoder", None)
        transformer_only = bool(getattr(policy, "transformer_actor_class_name", None))
        if cfg is None or context_encoder is None:
            raise ValueError("No transformer optimizer configuration or context encoder found")

        transformer_lr = self._get_cfg_value(cfg, "learning_rate", self.learning_rate)
        if transformer_only:
            self.learning_rate = transformer_lr

        # WARNING: if transformer_only is True, we are optimizing the actor parameters directly using transformer optimizer
        if transformer_only:
            encoder_params = list(policy.actor.parameters())
        else:
            encoder_params = list(context_encoder.parameters())
        if not encoder_params:
            raise ValueError("No encoder parameters found")

        encoder_param_ids = {id(param) for param in encoder_params}
        policy_params = [param for param in policy.parameters() if id(param) not in encoder_param_ids]
        if not policy_params:
            raise ValueError("No policy parameters found")

        policy_optimizer = optim.Adam(policy_params, lr=self.learning_rate)
        encoder_optimizer = self._build_transformer_optimizer(cfg, encoder_params)
        transformer_lr_scheduler = self._build_transformer_lr_scheduler(cfg, encoder_optimizer)

        self.optimizer = CompositeOptimizer(policy_optimizer, encoder_optimizer)
        self.transformer_optimizer = encoder_optimizer
        self.transformer_lr_scheduler = transformer_lr_scheduler
        self.policy_params = policy_params
        self.encoder_params = encoder_params
        self.transformer_max_grad_norm = self._get_cfg_value(cfg, "max_grad_norm", self.max_grad_norm)
        self.last_encoder_grad_norm = 0.0

    def _build_transformer_optimizer(self, cfg: Any, params: Iterable) -> optim.Optimizer:
        optimizer_class = self._get_cfg_value(cfg, "optimizer_class", "AdamW")
        optimizer_cls = getattr(optim, optimizer_class)
        learning_rate = self._get_cfg_value(cfg, "learning_rate", self.learning_rate)
        weight_decay = self._get_cfg_value(cfg, "weight_decay", 0.0)
        betas = self._get_cfg_value(cfg, "betas", (0.9, 0.99))
        eps = self._get_cfg_value(cfg, "eps", 1.0e-8)
        return optimizer_cls(params, lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps)

    def _build_transformer_lr_scheduler(self, cfg: Any, optimizer: optim.Optimizer) -> Any | None:
        schedule_name = self._get_cfg_value(cfg, "lr_schedule", None)
        if not schedule_name:
            return None
        warmup_steps = self._get_cfg_value(cfg, "lr_warmup_steps", 0)
        schedule_fn = eval(
            schedule_name,
            {
                "cosine_annealing_with_warmup": cosine_annealing_with_warmup,
                "linear_warmup": linear_warmup,
                "__builtins__": {},
            },
        )
        if schedule_name == "cosine_annealing_with_warmup":
            total_steps = self._resolve_lr_total_steps(warmup_steps)
            return schedule_fn(optimizer, warmup_steps, total_steps)
        return schedule_fn(optimizer, warmup_steps)

    def _resolve_lr_total_steps(self, warmup_steps: int) -> int:
        total_steps = self.num_learning_iterations
        assert total_steps is not None, "num_learning_iterations must be set"
        assert total_steps > warmup_steps, "num_learning_iterations must be greater than warmup_steps"
        return total_steps

    def _amp_ctx(self):
        # Small helper so we don’t duplicate logic everywhere
        if self.use_bf16_amp:
            return torch.cuda.amp.autocast(dtype=self.amp_dtype)
        # no-op context manager
        return torch.autograd.profiler.record_function("noop_amp_ctx")  # harmless, acts as context manager

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_encoder_grad_norm = 0.0
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.shape[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"]
                )
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # ---- forward + losses under BF16 autocast ----
            with self._amp_ctx():
                self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
                value_batch = self.policy.evaluate(
                    obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
                )
                mu_batch = self.policy.action_mean[:original_batch_size]
                sigma_batch = self.policy.action_std[:original_batch_size]
                entropy_batch = self.policy.entropy[:original_batch_size]

                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                if self.symmetry:
                    if not self.symmetry["use_data_augmentation"]:
                        data_augmentation_func = self.symmetry["data_augmentation_func"]
                        obs_batch, _ = data_augmentation_func(
                            obs=obs_batch, actions=None, env=self.symmetry["_env"]
                        )
                        num_aug = int(obs_batch.shape[0] / original_batch_size)

                    mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                    action_mean_orig = mean_actions_batch[:original_batch_size]
                    _, actions_mean_symm_batch = data_augmentation_func(
                        obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                    )

                    mse_loss = torch.nn.MSELoss()
                    symmetry_loss = mse_loss(
                        mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                    )
                    if self.symmetry["use_mirror_loss"]:
                        loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                    else:
                        symmetry_loss = symmetry_loss.detach()

                if self.rnd:
                    assert not self.rnd, "RND is not implemented/safe? for transformer PPO"
                    # predicted_embedding = self.rnd.predictor(rnd_state_batch)
                    # target_embedding = self.rnd.target(rnd_state_batch).detach()
                    # mseloss = torch.nn.MSELoss()
                    # rnd_loss = mseloss(predicted_embedding, target_embedding)

            # ---- KL/adaptive LR is inference-only; keep outside autocast ----
            action_distribution = getattr(self.policy, "action_distribution", "normal")
            if action_distribution != "categorical" and self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.policy_param_groups:
                        param_group["lr"] = self.learning_rate

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # if self.rnd:
            #     self.rnd_optimizer.zero_grad(set_to_none=True)  # type: ignore
            #     rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy_params, self.max_grad_norm)
            encoder_grad_norm = nn.utils.clip_grad_norm_(self.encoder_params, self.transformer_max_grad_norm)
            self.last_encoder_grad_norm = float(encoder_grad_norm)

            self.optimizer.step()
            # if self.rnd_optimizer:
            #     self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_encoder_grad_norm += self.last_encoder_grad_norm
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
        
        if self.transformer_lr_scheduler is not None:
            self.transformer_lr_scheduler.step()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_encoder_grad_norm /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "encoder_grad_norm": mean_encoder_grad_norm,
            "transformer_lr * 100": self.transformer_optimizer.param_groups[0]["lr"] * 100.0,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    @staticmethod
    def _get_cfg_value(cfg: Any, key: str, default: Any) -> Any:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)