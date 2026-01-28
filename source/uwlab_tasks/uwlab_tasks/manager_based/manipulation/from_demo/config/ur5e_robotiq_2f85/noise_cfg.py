from isaaclab.utils import configclass

@configclass
class EnvironmentNoiseCfg:
    """Configuration for the environment noise."""

    noise_frequency_distribution: str = "bernoulli"
    noise_magnitude_distribution: str = "normal"
    noise_object: str | None = "noise_both" # "noise_receptive", "noise_insertive", "noise_both", or None
    constant_noise: bool = False
    bernoulli_noise_prob: float = 0.6
    noise_switch_step: int = 20
    mean_noise_frequency: float = 0.5
    std_noise_frequency: float = 0.25
    mean_noise_magnitude: float = 0.0
    std_noise_magnitude: float = 2.0
    max_noise_frequency: float = 0.5
    min_noise_frequency: float = 0.0
    max_noise_magnitude: float = 0.75
    min_noise_magnitude: float = 0.0