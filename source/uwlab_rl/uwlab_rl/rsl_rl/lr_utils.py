import functools
import math
from typing import Any

import torch

__all__ = ["cosine_annealing_with_warmup", "linear_warmup", "build_lr_scheduler"]


# source:  https://gist.github.com/akshaychawla/86d938bc6346cf535dce766c83f743ce
def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.0
    """
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier


def _constant_warmup(iteration, warmup_iterations):
    """
    Linear warmup from 0 --> 1.0, then constant
    """
    multiplier = 1.0
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    return multiplier


def cosine_annealing_with_warmup(optimizer, warmup_steps, total_steps):
    _decay_func = functools.partial(
        _cosine_decay_warmup,
        warmup_iterations=warmup_steps,
        total_iterations=total_steps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler


def linear_warmup(optimizer, warmup_steps):
    _decay_func = functools.partial(
        _constant_warmup,
        warmup_iterations=warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler


def build_lr_scheduler(cfg: Any, optimizer: torch.optim.Optimizer):
    schedule_name = getattr(cfg, "lr_schedule", None) if not isinstance(cfg, dict) else cfg.get("lr_schedule")
    if not schedule_name:
        return None
    warmup_steps = getattr(cfg, "lr_warmup_steps", 0) if not isinstance(cfg, dict) else cfg.get("lr_warmup_steps", 0)
    schedule_map = {
        "cosine_annealing_with_warmup": cosine_annealing_with_warmup,
        "linear_warmup": linear_warmup,
    }
    if schedule_name not in schedule_map:
        raise ValueError(f"Unknown lr_schedule: {schedule_name}")
    schedule_fn = schedule_map[schedule_name]
    if schedule_name == "cosine_annealing_with_warmup":
        total_steps = getattr(cfg, "num_steps", None) if not isinstance(cfg, dict) else cfg.get("num_steps")
        if total_steps is None:
            raise ValueError("num_steps must be set for cosine_annealing_with_warmup.")
        if total_steps <= warmup_steps:
            raise ValueError("num_steps must be greater than lr_warmup_steps for cosine_annealing_with_warmup.")
        return schedule_fn(optimizer, warmup_steps, int(total_steps))
    return schedule_fn(optimizer, int(warmup_steps))
