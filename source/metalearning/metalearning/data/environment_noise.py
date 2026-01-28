from collections import deque

import torch
from torch.distributions import Normal, Uniform

class EnvironmentNoise:
    """Noise for the environment."""

    def __init__(self, cfg: dict, num_envs: int, env, device: str = "cuda"):
        self.cfg = cfg
        self.env = env
        self.device = torch.device(device)
        noise_frequency_dist_name = cfg.get("noise_frequency_distribution", "uniform")
        if noise_frequency_dist_name == "uniform":
            self.max_noise_frequency: float = cfg.get("max_noise_frequency", 1.0)
            self.min_noise_frequency: float = cfg.get("min_noise_frequency", 0.0)
            self.noise_frequency_distribution = Uniform(self.min_noise_frequency, self.max_noise_frequency)
        elif noise_frequency_dist_name == "normal":
            self.mean_noise_frequency: float = cfg.get("mean_noise_frequency", 0.5)
            self.std_noise_frequency: float = cfg.get("std_noise_frequency", 0.25)
            self.noise_frequency_distribution = Normal(self.mean_noise_frequency, self.std_noise_frequency)
        elif noise_frequency_dist_name == "bernoulli":
            self.bernoulli_noise_prob: float = cfg.get("bernoulli_noise_prob", 0.5)
            self.noise_frequency_distribution = None
        else:
            raise ValueError(f"Invalid noise frequency distribution: {noise_frequency_dist_name}")
        
        noise_magnitude_dist_name = cfg.get("noise_magnitude_distribution", "uniform")
        if noise_magnitude_dist_name == "uniform":
            self.max_noise_magnitude: float = cfg.get("max_noise_magnitude", 1.0)
            self.min_noise_magnitude: float = cfg.get("min_noise_magnitude", 0.0)
            self.noise_magnitude_distribution = Uniform(self.min_noise_magnitude, self.max_noise_magnitude)
        elif noise_magnitude_dist_name == "normal":
            self.mean_noise_magnitude: float = cfg.get("mean_noise_magnitude", 1.0)
            self.std_noise_magnitude: float = cfg.get("std_noise_magnitude", 0.5)
            self.noise_magnitude_distribution = Normal(self.mean_noise_magnitude, self.std_noise_magnitude)
        else:
            raise ValueError(f"Invalid noise magnitude distribution: {noise_magnitude_dist_name}")

        self.noise_object = cfg.get("noise_object", None)
        valid_noise_objects = {"noise_receptive", "noise_insertive", "noise_both", None}
        if self.noise_object not in valid_noise_objects:
            raise ValueError(f"Invalid noise_object: {self.noise_object}. Must be one of: {sorted(valid_noise_objects)}")
        self.constant_noise: bool = bool(cfg.get("constant_noise", False))
        self._noise_switch_step = int(cfg.get("noise_switch_step", 20))

        self.noise_prob = self._sample_noise_frequency(num_envs)
        self.noise_prob = self.noise_prob.clamp(0.0, 1.0)
        self._noise_magnitude_per_env = None
        self._noise_sign_per_env = None
        if self.constant_noise:
            self._noise_magnitude_per_env = self._sample_constant_magnitude(num_envs)
            self._noise_sign_per_env = self._sample_constant_signs(num_envs)
        self._stats_buffer_size = int(cfg.get("stats_buffer_size", 10000))
        self._resample_frequency_history = deque(maxlen=self._stats_buffer_size)
        self._noise_magnitude_history = deque(maxlen=self._stats_buffer_size)
        self._noise_application_count_history = deque(maxlen=self._stats_buffer_size)
        self._total_steps = 0
        self._total_noised_steps = 0
        self._per_env_noise_count = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self._resample_frequency_history.extend(self.noise_prob.detach().cpu().tolist())

    def step_action(self, action: torch.Tensor):
        """Noise the action"""
        if self.noise_object is not None:
            return action
        noise = torch.zeros_like(action)
        noised_env_mask = torch.rand_like(self.noise_prob) < self.noise_prob
        if noised_env_mask.any():
            env_ids = noised_env_mask.nonzero(as_tuple=False).squeeze(-1)
            noise_samples = self._sample_noise_magnitudes(noise[noised_env_mask].shape, env_ids)
            noise_signs = self._sample_noise_signs(noise_samples.shape, env_ids)
            noise[noised_env_mask] = noise_samples * noise_signs
            # Advanced indexing: collect per-env magnitudes from masked actions.
            per_env_magnitude = noise_samples.reshape(noise_samples.shape[0], -1).abs().mean(dim=1)
            self._noise_magnitude_history.extend(per_env_magnitude.detach().cpu().tolist())
            self._per_env_noise_count[noised_env_mask] += 1
            self._total_noised_steps += int(noised_env_mask.sum().item())
        self._total_steps += int(self.noise_prob.numel())
        return action + noise

    def apply_object_noise(self, target_pos: torch.Tensor, object_type: str | None) -> torch.Tensor:
        """Noise the target position based on object type."""
        if self.noise_object is None:
            return target_pos
        normalized = self._normalize_object_type(object_type)
        if not self._should_noise_object(normalized):
            return target_pos
        noise = torch.zeros_like(target_pos)
        noised_env_mask = torch.rand_like(self.noise_prob) < self.noise_prob
        noised_env_mask = self._apply_noise_both_switch(noised_env_mask, normalized)
        if noised_env_mask.any():
            env_ids = noised_env_mask.nonzero(as_tuple=False).squeeze(-1)
            noise_samples = self._sample_noise_magnitudes(noise[noised_env_mask].shape, env_ids)
            noise_signs = self._sample_noise_signs(noise_samples.shape, env_ids)
            noise[noised_env_mask] = noise_samples * noise_signs
            # Advanced indexing: only apply noise to x,y while keeping z intact.
            noise[noised_env_mask, 2] = 0.0
            # Advanced indexing: collect per-env magnitudes from masked positions.
            per_env_magnitude = noise_samples.reshape(noise_samples.shape[0], -1).abs().mean(dim=1)
            self._noise_magnitude_history.extend(per_env_magnitude.detach().cpu().tolist())
            self._per_env_noise_count[noised_env_mask] += 1
            self._total_noised_steps += int(noised_env_mask.sum().item())
        self._total_steps += int(self.noise_prob.numel())
        return target_pos + noise

    def resample_for_envs(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        assert env_ids.device == self.device
        self._noise_application_count_history.extend(self._per_env_noise_count[env_ids].detach().cpu().tolist())
        self._per_env_noise_count[env_ids] = 0
        new_freq = self._sample_noise_frequency(len(env_ids))
        new_freq = new_freq.clamp(0.0, 1.0)
        self.noise_prob[env_ids] = new_freq
        self._resample_frequency_history.extend(new_freq.detach().cpu().tolist())
        if self.constant_noise:
            if self._noise_magnitude_per_env is None:
                raise RuntimeError("Constant noise enabled without per-env magnitudes.")
            self._noise_magnitude_per_env[env_ids] = self._sample_constant_magnitude(len(env_ids))
            if self._noise_sign_per_env is None:
                raise RuntimeError("Constant noise enabled without per-env signs.")
            self._noise_sign_per_env[env_ids] = self._sample_constant_signs(len(env_ids))

    def get_statistics(self) -> dict:
        """Return aggregated statistics and buffers for logging."""
        avg_noise_per_reset = (
            float(sum(self._noise_application_count_history)) / max(len(self._noise_application_count_history), 1)
        )
        avg_noise_magnitude = (
            float(sum(self._noise_magnitude_history)) / max(len(self._noise_magnitude_history), 1)
        )
        noise_apply_rate = float(self._total_noised_steps) / max(self._total_steps, 1)
        stats = {
            "resample_frequency_history": list(self._resample_frequency_history),
            "noise_magnitude_history": list(self._noise_magnitude_history),
            "noise_application_count_history": list(self._noise_application_count_history),
            "avg_noise_applications_per_reset": avg_noise_per_reset,
            "avg_noise_magnitude": avg_noise_magnitude,
            "noise_apply_rate": noise_apply_rate,
            "current_noise_frequency_mean": float(self.noise_prob.mean().item()),
            "current_noise_frequency_std": float(self.noise_prob.std().item()),
        }
        self._reset_statistics()
        return stats

    def _reset_statistics(self) -> None:
        """Clear logging buffers and aggregates."""
        self._resample_frequency_history.clear()
        self._noise_magnitude_history.clear()
        self._noise_application_count_history.clear()
        self._total_steps = 0
        self._total_noised_steps = 0

    def _sample_noise_magnitudes(self, sample_shape: torch.Size, env_ids: torch.Tensor) -> torch.Tensor:
        if not self.constant_noise:
            return self.noise_magnitude_distribution.sample(sample_shape).to(self.device)
        if self._noise_magnitude_per_env is None:
            raise RuntimeError("Constant noise enabled without per-env magnitudes.")
        per_env = self._noise_magnitude_per_env[env_ids]
        expand_shape = (per_env.shape[0],) + (1,) * (len(sample_shape) - 1)
        # Advanced indexing: broadcast per-env magnitudes across tensor dims.
        return per_env.view(expand_shape).expand(sample_shape)

    def _sample_constant_magnitude(self, num_envs: int) -> torch.Tensor:
        return self.noise_magnitude_distribution.sample((num_envs, 1)).to(self.device).abs()

    def _sample_noise_signs(self, sample_shape: torch.Size, env_ids: torch.Tensor) -> torch.Tensor:
        if not self.constant_noise:
            return torch.randint(sample_shape, 2, device=self.device) * 2 - 1
        if self._noise_sign_per_env is None:
            raise RuntimeError("Constant noise enabled without per-env signs.")
        per_env = self._noise_sign_per_env[env_ids]
        expand_shape = (per_env.shape[0],) + (1,) * (len(sample_shape) - 1)
        # Advanced indexing: broadcast per-env signs across tensor dims.
        return per_env.view(expand_shape).expand(sample_shape)

    def _sample_constant_signs(self, num_envs: int) -> torch.Tensor:
        return torch.randint(0, 2, (num_envs, 1), device=self.device) * 2 - 1

    def _sample_noise_frequency(self, num_envs: int) -> torch.Tensor:
        if self.noise_frequency_distribution is not None:
            return self.noise_frequency_distribution.sample((num_envs,)).to(self.device)
        return torch.bernoulli(torch.full((num_envs,), self.bernoulli_noise_prob, device=self.device))

    def _normalize_object_type(self, object_type: str | None) -> str | None:
        if object_type is None:
            return None
        normalized = object_type.lower()
        if normalized.endswith("_object"):
            normalized = normalized[: -len("_object")]
        return normalized

    def _should_noise_object(self, normalized: str | None) -> bool:
        if normalized not in {"receptive", "insertive"}:
            return False
        if self.noise_object == "noise_both":
            return True
        if self.noise_object == "noise_receptive":
            return normalized == "receptive"
        return normalized == "insertive"

    def _apply_noise_both_switch(self, noised_env_mask: torch.Tensor, normalized: str | None) -> torch.Tensor:
        if self.noise_object != "noise_both":
            return noised_env_mask
        if normalized not in {"receptive", "insertive"}:
            return noised_env_mask
        episode_steps = self._get_episode_steps()
        if episode_steps is None:
            return noised_env_mask
        if normalized == "insertive":
            switch_mask = episode_steps < self._noise_switch_step
        else:
            switch_mask = episode_steps >= self._noise_switch_step
        return noised_env_mask & switch_mask

    def _get_episode_steps(self) -> torch.Tensor | None:
        episode_steps = getattr(self.env, "episode_length_buffer", None)
        if episode_steps is None:
            episode_steps = getattr(self.env, "episode_length_buf", None)
        if episode_steps is None:
            return None
        if episode_steps.device != self.device:
            episode_steps = episode_steps.to(self.device)
        return episode_steps