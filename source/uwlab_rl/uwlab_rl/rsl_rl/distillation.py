from __future__ import annotations

from typing import Any

import torch.optim as optim

from rsl_rl.algorithms.distillation import Distillation as BaseDistillation

from uwlab_rl.rsl_rl.lr_utils import cosine_annealing_with_warmup, linear_warmup


class Distillation(BaseDistillation):
    """Distillation algorithm with optional transformer optimizer schedule."""

    def __init__(
        self,
        policy,
        transformer_optimizer_cfg: Any | None = None,
        num_learning_iterations: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy, **kwargs)
        self.num_learning_iterations = num_learning_iterations
        self.transformer_lr_scheduler = None

        cfg = transformer_optimizer_cfg or getattr(policy, "transformer_optimizer_cfg", None)
        if cfg is None:
            return

        self.optimizer = self._build_transformer_optimizer(cfg, self.policy.parameters())
        self.transformer_lr_scheduler = self._build_transformer_lr_scheduler(cfg, self.optimizer)

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        if self.transformer_lr_scheduler is not None:
            self.transformer_lr_scheduler.step()
        return loss_dict

    def _build_transformer_optimizer(self, cfg: Any, params) -> optim.Optimizer:
        optimizer_class = self._get_cfg_value(cfg, "optimizer_class", "AdamW")
        optimizer_cls = getattr(optim, optimizer_class)
        learning_rate = self._get_cfg_value(cfg, "learning_rate", self.learning_rate)
        weight_decay = self._get_cfg_value(cfg, "weight_decay", 0.0)
        betas = self._get_cfg_value(cfg, "betas", (0.9, 0.99))
        eps = self._get_cfg_value(cfg, "eps", 1.0e-8)
        return optimizer_cls(params, lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps)

    def _build_transformer_lr_scheduler(self, cfg: Any, optimizer: optim.Optimizer):
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
        assert total_steps is not None, "num_learning_iterations must be set for cosine_annealing_with_warmup"
        assert total_steps > warmup_steps, "num_learning_iterations must be greater than warmup_steps"
        return total_steps

    @staticmethod
    def _get_cfg_value(cfg: Any, key: str, default: Any) -> Any:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)
