import torch

from isaaclab.envs import ManagerBasedEnv

from . import utils as mdp_utils



def cache_state_and_physics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    """Cache per-env state and physics into function attributes."""
    if not hasattr(cache_state_and_physics, "initialized"):
        cache_state_and_physics.state_buffer = {}
        cache_state_and_physics.physics_buffer = {}
        cache_state_and_physics.initialized = True

    if env_ids.numel() == 0:
        return

    current_state = env.scene.get_state(is_relative=True)
    if isinstance(cache_state_and_physics.state_buffer, dict) and cache_state_and_physics.state_buffer:
        mdp_utils.update_state_dict(cache_state_and_physics.state_buffer, current_state, env_ids)
    else:
        cache_state_and_physics.state_buffer = current_state

    current_physics = mdp_utils.collect_physics_for_envs(env, env_ids)
    if isinstance(cache_state_and_physics.physics_buffer, dict) and cache_state_and_physics.physics_buffer:
        mdp_utils.update_state_dict(cache_state_and_physics.physics_buffer, current_physics, env_ids)
    else:
        cache_state_and_physics.physics_buffer = current_physics


def resample_environment_noise(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Resample per-environment noise frequency on reset."""
    if env_ids.numel() == 0:
        return
    environment_noise = getattr(env, "environment_noise", None)
    if environment_noise is None:
        raise ValueError("Environment noise not found in environment.")
    environment_noise.resample_frequency(env_ids)