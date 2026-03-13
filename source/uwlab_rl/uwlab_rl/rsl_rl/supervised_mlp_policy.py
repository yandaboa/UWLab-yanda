from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal


class SupervisedMLPPolicy(nn.Module):
    """MLP policy for supervised action prediction with gaussian or mse losses."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        loss_type: str = "gaussian_nll",
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        if loss_type not in {"gaussian_nll", "mse"}:
            raise ValueError(f"Unsupported loss_type={loss_type}. Expected one of ['gaussian_nll', 'mse'].")

        layers: list[nn.Module] = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        output_dim = action_dim * 2 if loss_type == "gaussian_nll" else action_dim
        self.mean_head = nn.Linear(in_dim, output_dim)
        self.register_parameter("log_std_param", None)

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = list(hidden_dims)
        self.loss_type = str(loss_type)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.register_buffer("action_norm_mean", torch.zeros(action_dim, dtype=torch.float32))
        self.register_buffer("action_norm_std", torch.ones(action_dim, dtype=torch.float32))
        self.action_norm_mean: torch.Tensor
        self.action_norm_std: torch.Tensor

    def set_action_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set per-dimension action normalization buffers used in training and inference."""
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError("Action normalization mean/std must be 1D tensors.")
        if mean.shape != self.action_norm_mean.shape or std.shape != self.action_norm_std.shape:
            raise ValueError(
                f"Action normalization shape mismatch: expected {self.action_norm_mean.shape}, "
                f"got mean={mean.shape}, std={std.shape}."
            )
        self.action_norm_mean.copy_(mean.to(device=self.action_norm_mean.device, dtype=self.action_norm_mean.dtype))
        self.action_norm_std.copy_(std.to(device=self.action_norm_std.device, dtype=self.action_norm_std.dtype))

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        mean = self.action_norm_mean.to(device=actions.device, dtype=actions.dtype)
        std = self.action_norm_std.to(device=actions.device, dtype=actions.dtype)
        return (actions - mean) / std

    def unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        mean = self.action_norm_mean.to(device=actions.device, dtype=actions.dtype)
        std = self.action_norm_std.to(device=actions.device, dtype=actions.dtype)
        return actions * std + mean

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return action mean and optional per-dimension log-std."""
        features = self.backbone(states)
        outputs = self.mean_head(features)
        if self.loss_type != "gaussian_nll":
            return outputs, None
        mean, log_std = torch.chunk(outputs, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute supervised loss against normalized action targets."""
        target = self.normalize_actions(actions)
        mean, log_std = self(states)
        if self.loss_type == "mse":
            return torch.mean((target - mean) ** 2)
        if log_std is None:
            raise RuntimeError("Gaussian NLL loss requires log_std output.")
        return gaussian_nll(target, mean, log_std)

    def action_distribution(self, states: torch.Tensor) -> Independent:
        """Build a diagonal Gaussian action distribution in normalized action space."""
        if self.loss_type != "gaussian_nll":
            raise RuntimeError("Action distribution is only available when loss_type='gaussian_nll'.")
        mean, log_std = self(states)
        assert log_std is not None
        return self._build_action_distribution(mean, log_std)

    def _build_action_distribution(self, mean: torch.Tensor, log_std: torch.Tensor) -> Independent:
        std = torch.exp(log_std)
        return Independent(Normal(loc=mean, scale=std), 1)

    def act(self, states: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        """Predict unnormalized actions, optionally sampling for gaussian models."""
        mean, log_std = self(states)
        if log_std is None:
            action = mean
        elif stochastic:
            action = self._build_action_distribution(mean, log_std).sample()
        else:
            action = mean
        return self.unnormalize_actions(action)


def gaussian_nll(actions: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood of a diagonal Gaussian."""
    var = torch.exp(2.0 * log_std).clamp_min(1.0e-8)
    return F.gaussian_nll_loss(mean, actions, var, reduction="none").sum(dim=-1).mean()


def load_supervised_policy_checkpoint(
    checkpoint_path: str, device: torch.device
) -> tuple[SupervisedMLPPolicy, dict[str, Any]]:
    """Load a supervised MLP checkpoint with backwards compatibility."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    required_keys = ["state_dim", "action_dim", "hidden_dims", "model_state_dict"]
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint {checkpoint_path} is missing required keys: {missing}")

    loss_type = str(checkpoint.get("loss_type", "gaussian_nll"))
    model = SupervisedMLPPolicy(
        state_dim=int(checkpoint["state_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_dims=list(checkpoint["hidden_dims"]),
        loss_type=loss_type,
        log_std_min=float(checkpoint.get("log_std_min", -5.0)),
        log_std_max=float(checkpoint.get("log_std_max", 2.0)),
    ).to(device)

    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    missing_keys = set(incompatible.missing_keys)
    unexpected_keys = set(incompatible.unexpected_keys)
    allowed_missing = {"action_norm_mean", "action_norm_std"}
    allowed_missing.add("log_std_param")
    disallowed_missing = missing_keys - allowed_missing
    if disallowed_missing or unexpected_keys:
        raise RuntimeError(
            "Checkpoint incompatibility while loading supervised policy. "
            f"missing={sorted(disallowed_missing)} unexpected={sorted(unexpected_keys)}"
        )

    if "action_norm_mean" in checkpoint and "action_norm_std" in checkpoint:
        action_norm_mean = torch.as_tensor(checkpoint["action_norm_mean"], dtype=torch.float32)
        action_norm_std = torch.as_tensor(checkpoint["action_norm_std"], dtype=torch.float32)
        model.set_action_normalization(action_norm_mean, action_norm_std)

    model.eval()
    return model, checkpoint
