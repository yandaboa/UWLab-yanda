import torch

from isaaclab.envs import ManagerBasedEnv


def resample_environment_noise(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Resample per-environment noise frequency on reset."""
    if env_ids.numel() == 0:
        return
    environment_noise = getattr(env, "environment_noise", None)
    if environment_noise is None:
        return
    raw_num_similar_trajectories = getattr(env, "num_similar_trajectories", None)
    num_collected_episodes = getattr(env, "num_collected_episodes", None)
    if (
        isinstance(raw_num_similar_trajectories, int)
        and raw_num_similar_trajectories > 1
        and isinstance(num_collected_episodes, torch.Tensor)
    ):
        should_resample = (
            num_collected_episodes[env_ids].to(dtype=torch.long)
            == (raw_num_similar_trajectories - 1)
        )
        if not torch.any(should_resample):
            return
        environment_noise.resample_for_envs(env_ids[should_resample])
        return
    environment_noise.resample_for_envs(env_ids)