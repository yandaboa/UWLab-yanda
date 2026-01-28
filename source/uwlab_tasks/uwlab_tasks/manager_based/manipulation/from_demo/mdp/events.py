import torch

from isaaclab.envs import ManagerBasedEnv


def resample_environment_noise(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Resample per-environment noise frequency on reset."""
    if env_ids.numel() == 0:
        return
    environment_noise = getattr(env, "environment_noise", None)
    if environment_noise is not None:
        environment_noise.resample_for_envs(env_ids)
    # else:
        # raise ValueError("Environment noise not found in environment.")