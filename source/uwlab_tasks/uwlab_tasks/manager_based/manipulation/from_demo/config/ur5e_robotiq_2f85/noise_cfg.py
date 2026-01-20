from isaaclab.utils import configclass

@configclass
class EnvironmentNoiseCfg:
    """Configuration for the environment noise."""

    noise_frequency_distribution: str = "normal"
    noise_magnitude_distribution: str = "normal"
    mean_noise_frequency: float = 0.5
    std_noise_frequency: float = 0.25
    mean_noise_magnitude: float = 1.0
    std_noise_magnitude: float = 0.5