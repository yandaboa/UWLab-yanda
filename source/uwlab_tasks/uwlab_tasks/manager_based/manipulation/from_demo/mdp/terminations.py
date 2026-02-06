from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["end_of_demo"]


def end_of_demo(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when the episode length exceeds the demo length."""
    context = getattr(env, "context", None)
    if context is None:
        raise AttributeError("Env has no demo context. Expected env.context to be initialized.")
    demo_lengths = context.demo_obs_lengths  # [num_envs]
    end_idx = (demo_lengths - 1).clamp(min=0)
    return env.episode_length_buf.long() > end_idx
