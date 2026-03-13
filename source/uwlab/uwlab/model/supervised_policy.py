import torch
from torch import nn

class GaussianMLPPolicy(nn.Module):
    """MLP that predicts Gaussian mean with a global diagonal log-std parameter."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int], log_std_min: float, log_std_max: float):
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_param = nn.Parameter(torch.zeros(action_dim))

        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(states)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)
        log_std = log_std.unsqueeze(0).expand_as(mean)
        return mean, log_std


def gaussian_nll(actions: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood of diagonal Gaussian policy."""
    log_2pi = 1.8378770664093453  # log(2 * pi)
    inv_var = torch.exp(-2.0 * log_std)
    per_dim_nll = 0.5 * (((actions - mean) ** 2) * inv_var + 2.0 * log_std + log_2pi)
    return per_dim_nll.sum(dim=-1).mean()