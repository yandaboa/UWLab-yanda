from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.assets import retrieve_file_path

from .mdp import utils

ACTION_DISCRETIZATION_SPEC_FILENAME = "action_discretization_spec.json"

class DemoTrackingContext:
    """Demo context stored on the environment."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        params: dict[str, Any],
        num_envs: int | None = None,
        device: str | torch.device | None = None,
    ):
        self._env = env
        self.use_raw_states = bool(params.get("use_raw_states", False))
        self.pending_raw_env_ids: torch.Tensor | None = None
        self.pending_raw_timesteps: torch.Tensor | None = None
        self.start_timesteps: torch.Tensor = torch.zeros((num_envs,), device=device, dtype=torch.int64)
        resolved_num_envs = num_envs if num_envs is not None else env.num_envs
        resolved_device = device if device is not None else env.device
        episode_paths = params.get("episode_paths", [])
        download_dir = params.get("download_dir")
        if download_dir is not None and not isinstance(download_dir, str):
            download_dir = str(download_dir)
        if isinstance(episode_paths, str):
            episode_paths = [episode_paths]
        if not isinstance(episode_paths, Iterable) or not episode_paths:
            raise ValueError("DemoTrackingContext requires non-empty episode_paths.")
        self.episodes: list[dict[str, Any]] = []
        action_discretization_spec: dict[str, Any] | None = None
        for path in episode_paths:
            local_path = retrieve_file_path(str(path), download_dir=download_dir)
            if action_discretization_spec is None:
                action_discretization_spec = _load_action_discretization_spec(Path(local_path).parent)
            data = torch.load(local_path, map_location="cpu")
            if not isinstance(data, dict) or "episodes" not in data:
                raise ValueError(f"Expected EpisodeStorage format in {local_path}.")
            episodes = data["episodes"]
            if not episodes:
                raise ValueError(f"No episodes found in {local_path}.")
            for episode in episodes:
                cleaned = _strip_debug_obs(episode)
                if "states" not in cleaned or "physics" not in cleaned:
                    raise ValueError(f"Episode in {local_path} is missing states or physics.")
                self.episodes.append(cleaned)
        if not self.episodes:
            raise ValueError("DemoTrackingContext found no episodes to load.")
        self.action_discretization_spec = action_discretization_spec
        self.episode_indices = torch.full((resolved_num_envs,), -1, device=resolved_device, dtype=torch.long)
        device = torch.device(resolved_device)
        (
            self.demo_obs_max_len,
            self.demo_obs_shapes,
            self.demo_obs_term_order,
            self.demo_obs_concat_dim,
        ) = _infer_demo_obs_spec(self.episodes)
        # override the max length to 100
        self.demo_obs_max_len = 100
        self.demo_obs, self.demo_obs_dict = _allocate_demo_obs_buffers(
            resolved_num_envs,
            self.demo_obs_max_len,
            self.demo_obs_shapes,
            self.demo_obs_concat_dim,
            device,
        )
        self.demo_obs_lengths = torch.zeros((resolved_num_envs,), dtype=torch.int32, device=device)
        self.demo_action_shape = _infer_sequence_shape(self.episodes, "actions")
        self.demo_reward_shape = _infer_sequence_shape(self.episodes, "rewards")
        self.demo_actions = torch.zeros(
            (resolved_num_envs, self.demo_obs_max_len, *self.demo_action_shape), device=device
        )
        self.demo_rewards = torch.zeros(
            (resolved_num_envs, self.demo_obs_max_len, *self.demo_reward_shape), device=device
        )
        raw_noise_scale = params.get("state_noise_scale", 0.0)
        self.state_noise_scale = float(raw_noise_scale) if isinstance(raw_noise_scale, (int, float)) else 0.0
        self._physics_zero_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self.task_states_by_env: list[Any] = [None for _ in range(resolved_num_envs)]
        self.task_physics_by_env: list[Any] = [None for _ in range(resolved_num_envs)]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        if env_ids.numel() == 0:
            return
        episode_indices = torch.randint(
            0, len(self.episodes), (env_ids.numel(),), device=self._env.device
        )
        self.episode_indices[env_ids] = episode_indices
        selected = [self.episodes[idx] for idx in episode_indices.tolist()]
        padded_obs, padded_obs_dict, lengths = _pad_demo_obs_batch(
            [episode["obs"] for episode in selected],
            self.demo_obs_max_len,
            self.demo_obs_shapes,
            self.demo_obs_term_order,
            torch.device(self._env.device),
        )
        padded_actions = _pad_sequence_batch(
            [episode["actions"] for episode in selected],
            self.demo_obs_max_len,
            self.demo_action_shape,
            torch.device(self._env.device),
        )
        padded_rewards = _pad_sequence_batch(
            [episode["rewards"] for episode in selected],
            self.demo_obs_max_len,
            self.demo_reward_shape,
            torch.device(self._env.device),
        )
        for env_id, episode in zip(env_ids.tolist(), selected):
            self.task_states_by_env[env_id] = episode.get("states")
            self.task_physics_by_env[env_id] = episode.get("physics")
        states = _stack_episode_field([episode["states"] for episode in selected], device=self._env.device)
        states = _add_state_noise(states, self.state_noise_scale)
        physics = _stack_episode_field([episode["physics"] for episode in selected], device=None)
        physics = _expand_physics_for_envs(
            physics, self._env.num_envs, env_ids, self._physics_zero_cache
        )
        env_ids_index = env_ids.tolist()
        self._assign_demo_obs(env_ids_index, padded_obs, padded_obs_dict, lengths)
        self.demo_actions[env_ids_index] = padded_actions
        self.demo_rewards[env_ids_index] = padded_rewards
        multi_reset_manager = getattr(self._env, "multi_reset_manager", None)
        if self.use_raw_states:
            if multi_reset_manager is None:
                raise ValueError("Raw-state reset requires env.multi_reset_manager to be available.")
            raw_state_entries: list[dict[str, Any]] = []
            raw_timesteps: list[int] = []
            for episode in selected:
                raw_states = episode.get("raw_states")
                if not isinstance(raw_states, list) or not raw_states:
                    raise ValueError("Requested raw-state reset but episode has no raw_states.")
                sample_idx = int(torch.randint(0, len(raw_states), (1,), device=self._env.device).item())
                entry = raw_states[sample_idx]
                if not isinstance(entry, dict) or "state" not in entry or "timestep" not in entry:
                    raise ValueError("Invalid raw_states entry format.")
                raw_state_entries.append(entry["state"])
                raw_timesteps.append(int(entry["timestep"]))
            multi_reset_manager.load_raw_states(env_ids, raw_state_entries)
            self.pending_raw_env_ids = env_ids.clone()
            self.pending_raw_timesteps = torch.tensor(
                raw_timesteps, device=self._env.device, dtype=self._env.episode_length_buf.dtype
            )
        elif _is_multi_reset_state(states) and multi_reset_manager is not None:
            assert "multi_reset_task_id" in states and "multi_reset_state_index" in states
            task_ids = states["multi_reset_task_id"]
            state_indices = states["multi_reset_state_index"]
            multi_reset_manager.load_saved_states(env_ids, state_indices, task_ids=task_ids)
        else:
            self._env.scene.reset_to(states, env_ids=env_ids, is_relative=True)  # type: ignore[arg-type]
        utils.apply_physics_for_envs(self._env, env_ids, physics)

    def _assign_demo_obs(
        self,
        env_ids: list[int],
        padded_obs: torch.Tensor,
        padded_obs_dict: dict[str, torch.Tensor] | None,
        lengths: torch.Tensor,
    ) -> None:
        self.demo_obs[env_ids] = padded_obs
        if padded_obs_dict is not None and self.demo_obs_dict is not None:
            for key, value in padded_obs_dict.items():
                self.demo_obs_dict[key][env_ids] = value
        self.demo_obs_lengths[env_ids] = lengths


class FromDemoEnv(ManagerBasedRLEnv):
    """Manager-based env that stores demo context on the environment."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self.context_cfg = getattr(cfg, "context", None)
        if self.context_cfg is not None:
            context_num_envs = getattr(cfg.scene, "num_envs", None)
            context_device = getattr(cfg.sim, "device", None)
            self.context = self._build_demo_context(context_num_envs, context_device)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def _reset_idx(self, env_ids):
        """Run custom post-reset events after base reset."""
        super()._reset_idx(env_ids)
        # we set the timestep here because the env sets timestep to 0 after reset
        context = getattr(self, "context", None)
        if context is not None:
            pending_env_ids = getattr(context, "pending_raw_env_ids", None)
            pending_raw_timesteps = getattr(context, "pending_raw_timesteps", None)
            if pending_env_ids is not None and pending_raw_timesteps is not None:
                self.episode_length_buf[pending_env_ids] = pending_raw_timesteps
                context.start_timesteps[pending_env_ids] = pending_raw_timesteps
                context.pending_raw_env_ids = None
                context.pending_raw_timesteps = None
        # self.event_manager.apply(mode="post_reset", env_ids=env_ids)

    def _build_demo_context(
        self, num_envs: int | None, device: str | torch.device | None
    ) -> DemoTrackingContext | None:
        if self.context_cfg is None:
            return None
        if isinstance(self.context_cfg, dict):
            params = dict(self.context_cfg)
        else:
            params = {
                "episode_paths": getattr(self.context_cfg, "episode_paths", None),
                "state_noise_scale": getattr(self.context_cfg, "state_noise_scale", None),
                "download_dir": getattr(self.context_cfg, "download_dir", None),
                "use_raw_states": getattr(self.context_cfg, "use_raw_states", None),
            }
        return DemoTrackingContext(self, params, num_envs=num_envs, device=device)


def _strip_debug_obs(episode: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(episode)
    obs = cleaned.get("obs")
    if isinstance(obs, dict):
        obs = {
            key: value
            for key, value in obs.items()
            if key not in {"debug", "debug_obs"} and not key.startswith("debug/")
        }
        cleaned["obs"] = obs
    return cleaned


def _is_multi_reset_state(states: Any) -> bool:
    if not isinstance(states, dict):
        return False
    return "multi_reset_task_id" in states and "multi_reset_state_index" in states


def _load_action_discretization_spec(directory: Path) -> dict[str, Any] | None:
    spec_path = directory / ACTION_DISCRETIZATION_SPEC_FILENAME
    if not spec_path.exists():
        return None
    try:
        return json.loads(spec_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse action discretization spec: {spec_path}") from exc


def _stack_episode_field(values: list[Any], device: torch.device | str | None) -> Any:
    if not values:
        raise ValueError("Cannot stack empty episode fields.")
    first = values[0]
    if isinstance(first, dict):
        return {key: _stack_episode_field([value[key] for value in values], device) for key in first}
    if isinstance(first, torch.Tensor):
        stacked = torch.stack(values, dim=0)
        return stacked if device is None else stacked.to(device)
    raise TypeError(f"Unsupported episode field type: {type(first)}")


def _add_state_noise(states: Any, scale: float) -> Any:
    if scale <= 0.0:
        return states
    if isinstance(states, dict):
        return {key: _add_state_noise(value, scale) for key, value in states.items()}
    if isinstance(states, torch.Tensor):
        return states + torch.randn_like(states) * scale
    return states


def _expand_physics_for_envs(
    physics: Any,
    num_envs: int,
    env_ids: torch.Tensor,
    zero_cache: dict[tuple[Any, ...], torch.Tensor],
) -> Any:
    """Expand per-env physics tensors to full [num_envs, ...] buffers."""
    if isinstance(physics, dict):
        return {
            key: _expand_physics_for_envs(value, num_envs, env_ids, zero_cache)
            for key, value in physics.items()
        }
    if isinstance(physics, torch.Tensor):
        env_ids_device = env_ids.to(physics.device)
        key = (num_envs, *physics.shape[1:], str(physics.device), physics.dtype)
        full = zero_cache.get(key)
        if full is None or full.device != physics.device:
            full = torch.zeros(
                (num_envs, *physics.shape[1:]), device=physics.device, dtype=physics.dtype
            )
            zero_cache[key] = full
        else:
            full.zero_()
        if env_ids_device.numel() > 0:
            full.index_copy_(0, env_ids_device, physics)
        return full
    return physics


def _infer_demo_obs_spec(
    episodes: list[dict[str, Any]],
) -> tuple[int, dict[str, tuple[int, ...]] | tuple[int, ...], list[str] | None, int]:
    first_obs = episodes[0].get("obs")
    if isinstance(first_obs, dict):
        term_order = list(first_obs.keys())
        obs_shapes: dict[str, tuple[int, ...]] | tuple[int, ...] = {
            key: tuple(value.shape[1:]) for key, value in first_obs.items()
        }
        concat_dim = int(sum(math.prod(shape) for shape in obs_shapes.values()))
    elif isinstance(first_obs, torch.Tensor):
        term_order = None
        obs_shapes = tuple(first_obs.shape[1:])
        concat_dim = int(math.prod(obs_shapes))
    else:
        raise TypeError(f"Unsupported obs type: {type(first_obs)}")
    max_len = 0
    for episode in episodes:
        obs = episode.get("obs")
        if isinstance(obs, dict):
            length = int(next(iter(obs.values())).shape[0])
            if not set(obs.keys()) == set(term_order or []):
                raise ValueError("Episode obs keys must match across datasets.")
            if obs_shapes != {key: tuple(value.shape[1:]) for key, value in obs.items()}:
                raise ValueError("Episode obs shapes must match across datasets.")
        elif isinstance(obs, torch.Tensor):
            length = int(obs.shape[0])
            if obs_shapes != tuple(obs.shape[1:]):
                raise ValueError("Episode obs shapes must match across datasets.")
        else:
            raise TypeError(f"Unsupported obs type: {type(obs)}")
        max_len = max(max_len, length)
    print(f"Context observation terms' order: {term_order}")
    return max_len, obs_shapes, term_order, concat_dim


def _infer_sequence_shape(episodes: list[dict[str, Any]], key: str) -> tuple[int, ...]:
    first = episodes[0].get(key)
    if not isinstance(first, torch.Tensor):
        raise TypeError(f"Episode {key} must be a tensor.")
    shape = tuple(first.shape[1:])
    for episode in episodes:
        value = episode.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Episode {key} must be a tensor.")
        if tuple(value.shape[1:]) != shape:
            raise ValueError(f"Episode {key} shapes must match across datasets.")
    return shape


def _pad_sequence_batch(
    sequences: list[torch.Tensor], max_len: int, shape: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    padded = pad_sequence([sequence.to(device) for sequence in sequences], batch_first=True)
    if padded.shape[1] < max_len:
        pad = torch.zeros((padded.shape[0], max_len - padded.shape[1], *shape), device=device, dtype=padded.dtype)
        padded = torch.cat([padded, pad], dim=1)
    return padded


def _allocate_demo_obs_buffers(
    num_envs: int,
    max_len: int,
    obs_shapes: dict[str, tuple[int, ...]] | tuple[int, ...],
    concat_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    if isinstance(obs_shapes, dict):
        demo_obs_dict = {
            key: torch.zeros((num_envs, max_len, *shape), device=device) for key, shape in obs_shapes.items()
        }
        demo_obs = torch.zeros((num_envs, max_len, concat_dim), device=device)
        return demo_obs, demo_obs_dict
    return torch.zeros((num_envs, max_len, *obs_shapes), device=device), None


def _pad_demo_obs_batch(
    obs_list: list[dict[str, torch.Tensor] | torch.Tensor],
    max_len: int,
    obs_shapes: dict[str, tuple[int, ...]] | tuple[int, ...],
    term_order: list[str] | None,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None, torch.Tensor]:
    if isinstance(obs_shapes, dict):
        lengths = torch.tensor(
            [int(next(iter(obs.values())).shape[0]) for obs in obs_list if isinstance(obs, dict)],
            device=device,
            dtype=torch.int32,
        )
        stacked: dict[str, torch.Tensor] = {}
        for key in obs_shapes:
            sequences = []
            for obs in obs_list:
                if not isinstance(obs, dict):
                    raise TypeError("Expected dict obs for demo batch.")
                sequences.append(obs[key].to(device))
            padded = pad_sequence(sequences, batch_first=True)
            if padded.shape[1] < max_len:
                pad = torch.zeros(
                    (padded.shape[0], max_len - padded.shape[1], *obs_shapes[key]),
                    device=device,
                    dtype=padded.dtype,
                )
                padded = torch.cat([padded, pad], dim=1)
            stacked[key] = padded
        if term_order is None:
            term_order = list(obs_shapes.keys())
        padded_concat = _concatenate_demo_obs(stacked, term_order)
        return padded_concat, stacked, lengths
    lengths = torch.tensor(
        [int(obs.shape[0]) for obs in obs_list if isinstance(obs, torch.Tensor)],
        device=device,
        dtype=torch.int32,
    )
    sequences = []
    for obs in obs_list:
        if not isinstance(obs, torch.Tensor):
            raise TypeError("Expected tensor obs for demo batch.")
        sequences.append(obs.to(device))
    padded = pad_sequence(sequences, batch_first=True)
    if padded.shape[1] < max_len:
        pad = torch.zeros(
            (padded.shape[0], max_len - padded.shape[1], *obs_shapes),
            device=device,
            dtype=padded.dtype,
        )
        padded = torch.cat([padded, pad], dim=1)
    return padded, None, lengths


def _concatenate_demo_obs(stacked: dict[str, torch.Tensor], term_order: list[str]) -> torch.Tensor:
    sequences = []
    for key in term_order:
        value = stacked[key]
        # Flatten per-term features before concatenation.
        sequences.append(value.flatten(start_dim=2))
    if not sequences:
        raise ValueError("Cannot concatenate empty demo obs.")
    return torch.cat(sequences, dim=-1)
