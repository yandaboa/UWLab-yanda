from __future__ import annotations

from typing import Optional

import torch
from torch.distributions import Distribution, Categorical, constraints


class IndependentCategoricalDistribution(Distribution):
    """Independent categorical distributions per action dimension."""

    has_rsample = False
    arg_constraints = {"logits": constraints.real}
    _validate_args = False

    def __init__(
        self,
        action_bins: list[int] | tuple[int, ...],
        bin_values: list[torch.Tensor] | None = None,
        batch_shape: torch.Size = torch.Size(),
        event_shape: torch.Size = torch.Size(),
        validate_args: Optional[bool] = None,
    ):
        super().__init__(batch_shape, event_shape, validate_args)
        self.action_bins = tuple(int(value) for value in action_bins)
        self.bin_values = bin_values
        self._categoricals: list[Categorical] | None = None
        self._logits: list[torch.Tensor] | None = None

    def proba_distribution(self, logits: torch.Tensor) -> "IndependentCategoricalDistribution":
        if logits.ndim == 2:
            splits = torch.split(logits, self.action_bins, dim=-1)
        elif logits.ndim == 3 and logits.shape[1] == len(self.action_bins):
            splits = [logits[:, idx, : bins] for idx, bins in enumerate(self.action_bins)]
        else:
            raise ValueError("Expected logits shape [B, sum(bins)] or [B, num_actions, max_bins].")
        self._logits = list(splits)
        self._categoricals = [Categorical(logits=split) for split in splits]
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self._categoricals is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        if actions.dtype != torch.long:
            if self.bin_values is None:
                raise ValueError("Float actions require bin_values to map to indices.")
            indices = []
            for idx, values in enumerate(self.bin_values):
                distances = (actions[..., idx : idx + 1] - values.to(actions.device)).abs()
                indices.append(distances.argmin(dim=-1))
            actions = torch.stack(indices, dim=-1)
        if actions.shape[-1] != len(self._categoricals):
            raise ValueError("Actions must have shape [B, num_actions].")
        log_probs = [dist.log_prob(actions[..., idx]) for idx, dist in enumerate(self._categoricals)]
        return torch.stack(log_probs, dim=-1)

    def entropy(self) -> torch.Tensor:
        if self._categoricals is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        entropies = [dist.entropy() for dist in self._categoricals]
        return torch.stack(entropies, dim=-1)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        if self._categoricals is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        with torch.no_grad():
            return self.mode

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self.sample(sample_shape)

    @property
    def mean(self) -> torch.Tensor:
        if self._categoricals is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        means = []
        for idx, dist in enumerate(self._categoricals):
            probs = dist.probs
            values = self._resolve_bin_values(idx, probs.device, probs.dtype)
            means.append((probs * values).sum(dim=-1))
        return torch.stack(means, dim=-1)

    @property
    def stddev(self) -> torch.Tensor:
        if self._categoricals is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        stds = []
        for idx, dist in enumerate(self._categoricals):
            probs = dist.probs
            values = self._resolve_bin_values(idx, probs.device, probs.dtype)
            mean = (probs * values).sum(dim=-1, keepdim=True)
            var = (probs * (values - mean).pow(2)).sum(dim=-1)
            stds.append(torch.sqrt(var + 1e-8))
        return torch.stack(stds, dim=-1)

    @property
    def mode(self) -> torch.Tensor:
        if self._logits is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        indices = [logits.argmax(dim=-1) for logits in self._logits]
        return torch.stack(indices, dim=-1)

    @property
    def support(self) -> constraints.Constraint:
        return constraints.integer_interval(0, max(self.action_bins) - 1)

    def expand(self, batch_shape: torch.Size, _instance=None) -> "IndependentCategoricalDistribution":
        new = self._get_checked_instance(IndependentCategoricalDistribution, _instance)
        new.action_bins = self.action_bins
        new.bin_values = self.bin_values
        new._categoricals = self._categoricals
        new._logits = self._logits
        super(IndependentCategoricalDistribution, new).__init__(batch_shape, self._event_shape, validate_args=False)
        return new

    def _resolve_bin_values(self, idx: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.bin_values is not None:
            return self.bin_values[idx].to(device=device, dtype=dtype)
        bins = self.action_bins[idx]
        return torch.arange(bins, device=device, dtype=dtype)
