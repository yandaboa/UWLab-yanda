"""Utilities for logging demo-vs-rollout trajectories during training."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from isaaclab.envs import ManagerBasedEnv

from uwlab_tasks.manager_based.manipulation.from_demo.mdp import utils as from_demo_utils

from .rollout_pair_storage import demo_to_episode, rollout_to_episode
from .rollout_storage import RolloutStorage


def _flatten_debug_obs(debug_obs: Any) -> dict[str, torch.Tensor]:
    if isinstance(debug_obs, Mapping):
        return {f"debug/{key}": value for key, value in debug_obs.items()}
    if isinstance(debug_obs, torch.Tensor):
        return {"debug": debug_obs}
    return {}


def _update_demo_snapshot(
    demo_context: Any,
    env_ids: torch.Tensor,
    demo_obs_snapshot: dict[str, torch.Tensor] | None,
    demo_lengths_snapshot: torch.Tensor,
) -> None:
    if env_ids.numel() == 0:
        return
    env_ids_cpu = env_ids.detach().cpu()
    demo_lengths_snapshot[env_ids_cpu] = demo_context.demo_obs_lengths[env_ids].detach().cpu()
    if demo_obs_snapshot is None:
        return
    demo_obs_dict = demo_context.demo_obs_dict
    if demo_obs_dict is None:
        return
    for key, value in demo_obs_dict.items():
        demo_obs_snapshot[key][env_ids_cpu] = value[env_ids].detach().cpu()


class TrajectoryPairCollector:
    """Collect rollout + demo pairs for visualization."""

    def __init__(
        self,
        env: Any,
        obs_key: str = "debug/end_effector_pose",
        max_pairs_per_log: int = 1,
    ) -> None:
        self._env = env
        self._obs_key = obs_key
        self._max_pairs_per_log = max(1, int(max_pairs_per_log))

        self._device = env.unwrapped.device
        self._num_envs = env.unwrapped.num_envs
        self._collecting = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._pending_pairs: list[dict[str, Any]] = []
        self._listening = False

        self._rollout_storage = self._build_rollout_storage()
        self._demo_context = self._get_demo_context()
        self._demo_obs_snapshot, self._demo_lengths_snapshot = self._snapshot_demo_context()
        self._demo_indices = self._demo_context.episode_indices.clone()

    @property
    def is_listening(self) -> bool:
        return self._listening

    def arm(self) -> None:
        """Start listening for the next completed episodes."""
        self._listening = True

    def start_new_episodes(self, env_ids: torch.Tensor) -> None:
        """Begin collecting for new episodes after a reset."""
        if not self._listening or env_ids.numel() == 0:
            return
        self._rollout_storage.wipe_envs(env_ids)
        self._collecting[env_ids] = True

    def record_step(
        self,
        obs: Mapping[str, Any] | torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> list[dict[str, Any]] | None:
        """Record a step and return a batch if ready."""
        if not self._listening:
            return None
        env_ids = torch.nonzero(self._collecting, as_tuple=True)[0]
        if env_ids.numel() == 0:
            return None
        rollout_obs = self._build_rollout_obs(obs)
        self._rollout_storage.add_step(rollout_obs, actions, rewards, dones, env_ids=env_ids)
        done_env_ids = env_ids[(dones[env_ids] > 0).nonzero(as_tuple=True)[0]]
        if done_env_ids.numel() == 0:
            return None
        new_pairs = self._build_pairs(done_env_ids)
        if not new_pairs:
            return None
        self._pending_pairs.extend(new_pairs)
        self._collecting[done_env_ids] = False
        self._rollout_storage.wipe_envs(done_env_ids)
        _update_demo_snapshot(self._demo_context, done_env_ids, self._demo_obs_snapshot, self._demo_lengths_snapshot)
        self._demo_indices[done_env_ids] = self._demo_context.episode_indices[done_env_ids]
        if len(self._pending_pairs) >= self._max_pairs_per_log:
            batch = self._pending_pairs[: self._max_pairs_per_log]
            self._pending_pairs = []
            self._listening = False
            return batch
        return None

    def _build_rollout_storage(self) -> RolloutStorage:
        obs_buf = getattr(self._env.unwrapped, "obs_buf", None)
        obs_key = self._obs_key
        debug_key = obs_key.split("/", 1)[1] if obs_key.startswith("debug/") else obs_key
        if isinstance(obs_buf, Mapping):
            debug_obs = obs_buf.get("debug")
            if isinstance(debug_obs, Mapping) and debug_key in debug_obs:
                pose_obs = debug_obs[debug_key]
            elif obs_key in obs_buf:
                pose_obs = obs_buf[obs_key]
            else:
                raise KeyError(f"Unable to find '{obs_key}' in obs_buf for trajectory logging.")
        else:
            raise KeyError("Trajectory logging requires debug observations in obs_buf.")
        obs_shape = {obs_key: tuple(pose_obs.shape[1:])}
        action_space = getattr(self._env.unwrapped, "single_action_space", self._env.action_space)
        action_shape = from_demo_utils.extract_action_shape(action_space, num_envs=self._num_envs)
        max_steps = getattr(self._env.unwrapped, "max_episode_length", 0)
        return RolloutStorage(
            num_envs=self._num_envs,
            max_steps=max_steps,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=self._device,
        )

    def _get_demo_context(self) -> Any:
        manager_env = self._env.unwrapped
        if not isinstance(manager_env, ManagerBasedEnv):
            raise RuntimeError("Trajectory logging requires a ManagerBasedEnv.")
        demo_context = getattr(manager_env, "context", None)
        if demo_context is None:
            raise RuntimeError("Trajectory logging requires a demo context on the environment.")
        return demo_context

    def _snapshot_demo_context(self) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor]:
        demo_obs_snapshot = (
            {key: value.detach().cpu().clone() for key, value in self._demo_context.demo_obs_dict.items()}
            if self._demo_context.demo_obs_dict is not None
            else None
        )
        demo_lengths_snapshot = self._demo_context.demo_obs_lengths.detach().cpu().clone()
        return demo_obs_snapshot, demo_lengths_snapshot

    def _build_rollout_obs(self, obs: Mapping[str, Any] | torch.Tensor) -> dict[str, torch.Tensor]:
        if isinstance(obs, Mapping):
            if self._obs_key in obs:
                return {self._obs_key: obs[self._obs_key]}
            debug_obs = obs.get("debug") if isinstance(obs, Mapping) else None
            debug_flat = _flatten_debug_obs(debug_obs)
            if self._obs_key in debug_flat:
                return {self._obs_key: debug_flat[self._obs_key]}
        raise KeyError(f"Unable to extract '{self._obs_key}' from rollout observations.")

    def _build_pairs(self, done_env_ids: torch.Tensor) -> list[dict[str, Any]]:
        rollouts = self._rollout_storage.get_rollouts(done_env_ids)
        pairs: list[dict[str, Any]] = []
        done_env_list = done_env_ids.detach().cpu().tolist()
        for rollout_idx, env_id in enumerate(done_env_list):
            rollout_episode = rollout_to_episode(rollouts, rollout_idx, env_id)
            demo_index = int(self._demo_indices[env_id].item())
            demo_episode = demo_to_episode(self._demo_context, demo_index, env_id)
            length = int(self._demo_lengths_snapshot[env_id].item())
            if self._demo_obs_snapshot is None:
                raise RuntimeError("Expected demo observation dict for trajectory logging.")
            context_obs_dict = {
                key: value[env_id, :length].detach().cpu().clone()
                for key, value in self._demo_obs_snapshot.items()
            }
            demo_episode["obs"] = context_obs_dict
            demo_episode["length"] = length
            pairs.append({"context": demo_episode, "rollout": rollout_episode})
        return pairs
