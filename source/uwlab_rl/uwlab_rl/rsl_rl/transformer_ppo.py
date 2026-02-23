from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage.rollout_storage import RolloutStorage
from tensordict import TensorDict
from uwlab_rl.rsl_rl.lr_utils import cosine_annealing_with_warmup, linear_warmup
from uwlab_rl.rsl_rl.discrete_action_rollout_storage import DiscreteActionRolloutStorage

class CompositeOptimizer:
    """Proxy optimizer that forwards to a named set of optimizers."""

    def __init__(self, optimizers: dict[str, optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("CompositeOptimizer requires at least one optimizer.")
        self.optimizers = dict(optimizers)

    def zero_grad(self, set_to_none: bool = False) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable | None = None) -> dict[str, Any]:
        results = {}
        for name, optimizer in self.optimizers.items():
            results[name] = optimizer.step(closure=closure)
        return results

    def iter_param_groups(self, keys: Iterable[str] | None = None) -> Iterable[dict[str, Any]]:
        if keys is None:
            for optimizer in self.optimizers.values():
                yield from optimizer.param_groups
            return
        for key in keys:
            optimizer = self.optimizers.get(key)
            if optimizer is None:
                continue
            yield from optimizer.param_groups

    def set_lr(self, lr: float, keys: Iterable[str] | None = None) -> None:
        for param_group in self.iter_param_groups(keys):
            param_group["lr"] = lr

    def get_lr(self, key: str) -> float:
        optimizer = self.optimizers[key]
        return optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict[str, Any]:
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if isinstance(state_dict, dict) and all(isinstance(v, dict) for v in state_dict.values()):
            if set(state_dict.keys()) == set(self.optimizers.keys()):
                for name, opt_state in state_dict.items():
                    self.optimizers[name].load_state_dict(opt_state)
                return
            legacy_map = {"policy": "policy", "encoder": "transformer"}
            if set(state_dict.keys()).issubset(legacy_map.keys()):
                for legacy_key, opt_state in state_dict.items():
                    target_key = legacy_map[legacy_key]
                    if target_key in self.optimizers:
                        self.optimizers[target_key].load_state_dict(opt_state)
                return
        if len(self.optimizers) == 1:
            next(iter(self.optimizers.values())).load_state_dict(state_dict)
            return
        raise ValueError("State dict does not match CompositeOptimizer keys.")


class PPOWithLongContext(PPO):
    """PPO variant that optionally uses a dedicated transformer optimizer.

    Learning-rate behavior:
    - If a transformer optimizer is configured, it updates transformer params while the
      policy optimizer updates the remaining params.
    - If a transformer LR scheduler is provided, it governs the transformer LR; otherwise,
      adaptive KL-based LR is applied to both policy and transformer optimizers.
    - If no transformer optimizer is configured, a single optimizer covers all params and
      adaptive KL-based LR applies across the whole model.
    """

    def __init__(
        self,
        policy,
        transformer_optimizer_cfg: Any | None = None,
        num_learning_iterations: int | None = None,
        num_minibatches_per_update: int = 1,
        use_bf16_amp: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy, **kwargs)
        self.num_learning_iterations = num_learning_iterations
        self.num_minibatches_per_update = int(num_minibatches_per_update)
        if self.num_minibatches_per_update < 1:
            raise ValueError("num_minibatches_per_update must be >= 1.")
        if getattr(policy, "action_distribution", "normal") == "categorical":
            self.transition = DiscreteActionRolloutStorage.Transition()

        # ---- BF16 AMP toggle ----
        # Enable only on CUDA; BF16 autocast is CUDA-only in torch.cuda.amp
        self.use_bf16_amp = bool(use_bf16_amp and torch.cuda.is_available())
        self.amp_dtype = torch.bfloat16

        cfg = transformer_optimizer_cfg or getattr(policy, "transformer_optimizer_cfg", None)
        context_encoder = getattr(policy, "context_encoder", None)
        transformer_only = bool(getattr(policy, "context_token_layout", None))
        use_transformer_optimizer = cfg is not None and (context_encoder is not None or transformer_only)

        if use_transformer_optimizer:
            transformer_lr = self._get_cfg_value(cfg, "learning_rate", self.learning_rate)
            if transformer_only:
                self.learning_rate = transformer_lr

            # WARNING: if transformer_only is True, we are optimizing the actor parameters directly using transformer optimizer
            if transformer_only:
                transformer_params = list(policy.actor.parameters())
            else:
                transformer_params = list(context_encoder.parameters())
            if not transformer_params:
                raise ValueError("No transformer parameters found")

            transformer_param_ids = {id(param) for param in transformer_params}
            policy_params = [param for param in policy.parameters() if id(param) not in transformer_param_ids]
            if not policy_params:
                raise ValueError("No policy parameters found")

            policy_optimizer = optim.Adam(policy_params, lr=self.learning_rate)
            transformer_optimizer = self._build_transformer_optimizer(cfg, transformer_params)
            transformer_lr_scheduler = self._build_transformer_lr_scheduler(cfg, transformer_optimizer)

            self.optimizer = CompositeOptimizer(
                {
                    "policy": policy_optimizer,
                    "transformer": transformer_optimizer,
                }
            )
            self.transformer_optimizer = transformer_optimizer
            self.transformer_lr_scheduler = transformer_lr_scheduler
            self.transformer_learning_rate = transformer_lr
            self.use_transformer_adaptive_lr = transformer_lr_scheduler is None
            self.policy_params = policy_params
            self.encoder_params = transformer_params
            self.transformer_max_grad_norm = self._get_cfg_value(cfg, "max_grad_norm", self.max_grad_norm)
        else:
            policy_params = list(policy.parameters())
            if not policy_params:
                raise ValueError("No policy parameters found")
            policy_optimizer = optim.Adam(policy_params, lr=self.learning_rate)
            self.optimizer = CompositeOptimizer({"policy": policy_optimizer})
            self.transformer_optimizer = None
            self.transformer_lr_scheduler = None
            self.transformer_learning_rate = None
            self.use_transformer_adaptive_lr = False
            self.policy_params = policy_params
            self.encoder_params = []
            self.transformer_max_grad_norm = self.max_grad_norm

        self.last_encoder_grad_norm = 0.0
        self.log_attention_entropy = bool(getattr(policy, "log_attention_entropy", False))
        self.attention_entropy_interval = int(getattr(policy, "attention_entropy_interval", 0) or 0)

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        if getattr(self.policy, "action_distribution", "normal") == "categorical":
            action_bins = getattr(self.policy, "action_bins", None)
            if not action_bins:
                raise ValueError("Categorical policy must define action_bins.")
            action_logits_shape = (int(sum(action_bins)),)
            self.storage = DiscreteActionRolloutStorage(
                training_type,
                num_envs,
                num_transitions_per_env,
                obs,
                actions_shape,
                self.device,
                action_logits_shape=action_logits_shape,
            )
            return
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        if getattr(self.policy, "action_distribution", "normal") == "categorical":
            self.transition.action_logits = self.policy.action_logits.detach()
            self.transition.action_mean = None
            self.transition.action_sigma = None
        else:
            self.transition.action_mean = self.policy.action_mean.detach()
            self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

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

    def _apply_adaptive_lr(self, current_lr: float, kl_mean: torch.Tensor) -> float:
        new_lr = current_lr
        if kl_mean > self.desired_kl * 2.0:
            new_lr = max(1e-6, current_lr / 1.25)
        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            new_lr = min(1e-2, current_lr * 1.25)
        return new_lr

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

        num_updates = self.num_learning_epochs * self.num_mini_batches
        accumulation_steps = self.num_minibatches_per_update
        accumulated_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch_idx, batch in enumerate(generator, start=1):
            if len(batch) == 11:
                (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    old_action_logits_batch,
                    hid_states_batch,
                    masks_batch,
                ) = batch
            else:
                (
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
                ) = batch
                old_action_logits_batch = None
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
                if old_action_logits_batch is not None:
                    old_action_logits_batch = old_action_logits_batch.repeat(num_aug, 1)

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
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if action_distribution == "categorical":
                        if old_action_logits_batch is None:
                            raise RuntimeError("Missing old_action_logits_batch for categorical KL.")
                        old_logits = old_action_logits_batch[:original_batch_size]
                        new_logits = self.policy.action_logits[:original_batch_size].detach()
                        action_bins = getattr(self.policy, "action_bins", None)
                        if not action_bins:
                            raise RuntimeError("Categorical policy must define action_bins.")
                        old_splits = torch.split(old_logits, action_bins, dim=-1)
                        new_splits = torch.split(new_logits, action_bins, dim=-1)
                        kl_parts = []
                        for old_split, new_split in zip(old_splits, new_splits):
                            old_log_probs = torch.log_softmax(old_split, dim=-1)
                            new_log_probs = torch.log_softmax(new_split, dim=-1)
                            old_probs = torch.softmax(old_split, dim=-1)
                            kl_parts.append((old_probs * (old_log_probs - new_log_probs)).sum(dim=-1))
                        kl = torch.stack(kl_parts, dim=-1).sum(dim=-1)
                    else:
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
                        self.learning_rate = self._apply_adaptive_lr(self.learning_rate, kl_mean)
                        if self.use_transformer_adaptive_lr and self.transformer_learning_rate is not None:
                            self.transformer_learning_rate = self._apply_adaptive_lr(
                                self.transformer_learning_rate, kl_mean
                            )

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                        if self.use_transformer_adaptive_lr and self.transformer_learning_rate is not None:
                            transformer_lr_tensor = torch.tensor(
                                self.transformer_learning_rate, device=self.device
                            )
                            torch.distributed.broadcast(transformer_lr_tensor, src=0)
                            self.transformer_learning_rate = transformer_lr_tensor.item()

                    self.optimizer.set_lr(self.learning_rate, keys=["policy"])
                    if self.use_transformer_adaptive_lr and self.transformer_learning_rate is not None:
                        self.optimizer.set_lr(self.transformer_learning_rate, keys=["transformer"])

            (loss / accumulation_steps).backward()
            # if self.rnd:
            #     self.rnd_optimizer.zero_grad(set_to_none=True)  # type: ignore
            #     rnd_loss.backward()
            accumulated_batches += 1
            should_step_optimizer = accumulated_batches >= accumulation_steps or batch_idx == num_updates
            if should_step_optimizer:
                if accumulated_batches < accumulation_steps:
                    grad_scale = accumulation_steps / accumulated_batches
                    for parameter in self.policy.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(grad_scale)

                if self.is_multi_gpu:
                    self.reduce_parameters()

                nn.utils.clip_grad_norm_(self.policy_params, self.max_grad_norm)
                if self.encoder_params:
                    encoder_grad_norm = nn.utils.clip_grad_norm_(
                        self.encoder_params, self.transformer_max_grad_norm
                    )
                    self.last_encoder_grad_norm = float(encoder_grad_norm)
                else:
                    self.last_encoder_grad_norm = 0.0

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
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
            "transformer_lr * 100": (
                self.transformer_optimizer.param_groups[0]["lr"] * 100.0
                if self.transformer_optimizer is not None
                else self.optimizer.get_lr("policy") * 100.0
            ),
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