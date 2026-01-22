from collections import deque

import torch
from torch.distributions import Normal, Uniform

class EnvironmentNoise:
    """Noise for the environment."""

    def __init__(self, cfg: dict, num_envs: int, device: str = "cuda"):
        self.cfg = cfg
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
        
        self.noise_prob = self.noise_frequency_distribution.sample((num_envs,)).to(self.device)
        self.noise_prob = self.noise_prob.clamp(0.0, 1.0)
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
        noise = torch.zeros_like(action)
        noised_env_mask = torch.rand_like(self.noise_prob) < self.noise_prob
        if noised_env_mask.any():
            noise_samples = self.noise_magnitude_distribution.sample(noise[noised_env_mask].shape).to(self.device)
            negative_or_positive_samples = torch.randint_like(noise_samples, 2) * 2 - 1
            noise[noised_env_mask] = noise_samples * negative_or_positive_samples
            # Advanced indexing: collect per-env magnitudes from masked actions.
            per_env_magnitude = noise_samples.reshape(noise_samples.shape[0], -1).abs().mean(dim=1)
            self._noise_magnitude_history.extend(per_env_magnitude.detach().cpu().tolist())
            self._per_env_noise_count[noised_env_mask] += 1
            self._total_noised_steps += int(noised_env_mask.sum().item())
        self._total_steps += int(self.noise_prob.numel())
        return action + noise

    def resample_frequency(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        assert env_ids.device == self.device
        self._noise_application_count_history.extend(self._per_env_noise_count[env_ids].detach().cpu().tolist())
        self._per_env_noise_count[env_ids] = 0
        new_freq = self.noise_frequency_distribution.sample((len(env_ids),)).to(self.device)
        new_freq = new_freq.clamp(0.0, 1.0)
        self.noise_prob[env_ids] = new_freq
        self._resample_frequency_history.extend(new_freq.detach().cpu().tolist())

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