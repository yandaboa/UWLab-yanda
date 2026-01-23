from __future__ import annotations

from typing import Any, Callable, Iterable

import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from tensordict import TensorDict

class CompositeOptimizer:
    """Proxy optimizer that forwards to policy and encoder optimizers."""

    def __init__(self, policy_optimizer: optim.Optimizer, encoder_optimizer: optim.Optimizer) -> None:
        self.policy_optimizer = policy_optimizer
        self.encoder_optimizer = encoder_optimizer

    @property
    def param_groups(self) -> list[dict[str, Any]]:
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

    def __init__(self, policy, transformer_optimizer_cfg: Any | None = None, **kwargs: Any) -> None:
        super().__init__(policy, **kwargs)
        cfg = transformer_optimizer_cfg or getattr(policy, "transformer_optimizer_cfg", None)
        context_encoder = getattr(policy, "context_encoder", None)
        if cfg is None or context_encoder is None:
            raise ValueError("No transformer optimizer configuration or context encoder found")
        encoder_params = list(context_encoder.parameters())
        if not encoder_params:
            raise ValueError("No encoder parameters found")
        encoder_param_ids = {id(param) for param in encoder_params}
        policy_params = [param for param in policy.parameters() if id(param) not in encoder_param_ids]
        if not policy_params:
            raise ValueError("No policy parameters found")
        policy_optimizer = optim.Adam(policy_params, lr=self.learning_rate)
        encoder_optimizer = self._build_transformer_optimizer(cfg, encoder_params)

        # we assemble the two into a single optimizer for convenience + so rsl_rl packages can use it directly
        self.optimizer = CompositeOptimizer(policy_optimizer, encoder_optimizer)
        self.transformer_optimizer = encoder_optimizer

    def _build_transformer_optimizer(self, cfg: Any, params: Iterable) -> optim.Optimizer:
        optimizer_class = self._get_cfg_value(cfg, "optimizer_class", "AdamW")
        optimizer_cls = getattr(optim, optimizer_class)
        learning_rate = self._get_cfg_value(cfg, "learning_rate", self.learning_rate)
        weight_decay = self._get_cfg_value(cfg, "weight_decay", 0.0)
        betas = self._get_cfg_value(cfg, "betas", (0.9, 0.99))
        eps = self._get_cfg_value(cfg, "eps", 1.0e-8)
        return optimizer_cls(params, lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps)

    @staticmethod
    def _get_cfg_value(cfg: Any, key: str, default: Any) -> Any:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)
