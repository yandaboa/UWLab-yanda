from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch.distributions import Distribution, Categorical, constraints

from uwlab_rl.rsl_rl.discrete_action_utils import actions_to_indices


class IndependentCategoricalDistribution(Distribution):
    """Independent categorical distributions per action dimension.

    Conventions (typical for rsl_rl PPO):
      - log_prob(actions): [B, 1]
      - entropy():         [B]
      - sample():          [B, K]
      - mode:              [B, K]
    """

    has_rsample = False
    arg_constraints = {"logits": constraints.real}
    _validate_args = False

    def __init__(
        self,
        action_bins: Sequence[int],
        bin_values: Optional[Sequence[torch.Tensor]] = None,
        return_values: bool = False,
        batch_shape: torch.Size = torch.Size(),
        event_shape: torch.Size = torch.Size(),
        validate_args: Optional[bool] = None,
    ):
        super().__init__(batch_shape, event_shape, validate_args)
        self.action_bins = tuple(int(b) for b in action_bins)
        self.bin_values = list(bin_values) if bin_values is not None else None
        self.return_values = bool(return_values)

        self._categoricals: list[Categorical] | None = None
        self._logits: list[torch.Tensor] | None = None
        self._B: int | None = None
        self._K: int = len(self.action_bins)

    def proba_distribution(self, logits: torch.Tensor) -> "IndependentCategoricalDistribution":
        # logits: [B, sum(bins)] OR [B, K, max_bins]
        assert logits.ndim in (2, 3), f"logits must be 2D or 3D, got {tuple(logits.shape)}"

        if logits.ndim == 2:
            expected = sum(self.action_bins)
            assert logits.shape == (logits.shape[0], expected), (
                f"logits must be [B, sum(bins)={expected}], got {tuple(logits.shape)}"
            )
            splits = torch.split(logits, list(self.action_bins), dim=-1)
        else:
            assert logits.shape[1] == self._K, f"logits must be [B, K={self._K}, max_bins], got {tuple(logits.shape)}"
            assert logits.shape[2] >= max(self.action_bins), (
                f"logits.shape[2] must be >= max(bins)={max(self.action_bins)}, got {logits.shape[2]}"
            )
            splits = [logits[:, i, :b] for i, b in enumerate(self.action_bins)]

        self._B = int(logits.shape[0])
        self._logits = list(splits)
        self._categoricals = [Categorical(logits=s) for s in splits]
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self._categoricals is not None and self._B is not None, "Call proba_distribution first."

        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        assert actions.ndim == 2, f"actions must be [B,K] (or [B]), got {tuple(actions.shape)}"
        assert actions.shape[0] == self._B and actions.shape[1] == self._K, (
            f"actions must be [B={self._B}, K={self._K}], got {tuple(actions.shape)}"
        )

        idx = self._actions_to_indices(actions)  # [B,K] long
        assert idx.shape == (self._B, self._K), f"indices must be [B,K], got {tuple(idx.shape)}"

        per_dim = [dist.log_prob(idx[..., i]) for i, dist in enumerate(self._categoricals)]  # each [B]
        out = torch.stack(per_dim, dim=-1).sum(dim=-1, keepdim=True)  # [B,1]
        assert out.shape == (self._B, 1), f"log_prob must return [B,1], got {tuple(out.shape)}"
        return out

    def entropy(self) -> torch.Tensor:
        assert self._categoricals is not None and self._B is not None, "Call proba_distribution first."
        ent = torch.stack([dist.entropy() for dist in self._categoricals], dim=-1)  # [B,K]
        assert ent.shape[0] == self._B, f"entropy must return [B, K], got {tuple(ent.shape)}"
        return ent

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        assert self._categoricals is not None and self._B is not None, "Call proba_distribution first."
        assert sample_shape == torch.Size(), f"sample_shape not supported; expected empty, got {sample_shape}"

        with torch.no_grad():
            idx = torch.stack([dist.sample() for dist in self._categoricals], dim=-1)  # [B,K]
            out = self._maybe_map_to_values(idx)
            assert out.shape == (self._B, self._K), f"sample must return [B,K], got {tuple(out.shape)}"
            return out

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self.sample(sample_shape)

    @property
    def mode(self) -> torch.Tensor:
        assert self._logits is not None and self._B is not None, "Call proba_distribution first."
        idx = torch.stack([l.argmax(dim=-1) for l in self._logits], dim=-1)  # [B,K]
        out = self._maybe_map_to_values(idx)
        assert out.shape == (self._B, self._K), f"mode must return [B,K], got {tuple(out.shape)}"
        return out

    @property
    def mean(self) -> torch.Tensor:
        assert self._categoricals is not None and self._B is not None, "Call proba_distribution first."
        means = []
        for i, dist in enumerate(self._categoricals):
            probs = dist.probs  # [B,bins_i]
            values = self._resolve_bin_values(i, probs.device, probs.dtype)  # [bins_i]
            means.append((probs * values).sum(dim=-1))  # [B]
        out = torch.stack(means, dim=-1)  # [B,K]
        assert out.shape == (self._B, self._K), f"mean must return [B,K], got {tuple(out.shape)}"
        return out

    @property
    def stddev(self) -> torch.Tensor:
        assert self._categoricals is not None and self._B is not None, "Call proba_distribution first."
        stds = []
        for i, dist in enumerate(self._categoricals):
            probs = dist.probs
            values = self._resolve_bin_values(i, probs.device, probs.dtype)  # [bins_i]
            mean = (probs * values).sum(dim=-1, keepdim=True)  # [B,1]
            var = (probs * (values - mean).pow(2)).sum(dim=-1)  # [B]
            stds.append(torch.sqrt(var + 1e-8))
        out = torch.stack(stds, dim=-1)  # [B,K]
        assert out.shape == (self._B, self._K), f"stddev must return [B,K], got {tuple(out.shape)}"
        return out

    @property
    def support(self) -> constraints.Constraint:
        return constraints.integer_interval(0, max(self.action_bins) - 1)

    def expand(self, batch_shape: torch.Size, _instance=None) -> "IndependentCategoricalDistribution":
        new = self._get_checked_instance(IndependentCategoricalDistribution, _instance)
        new.action_bins = self.action_bins
        new.bin_values = self.bin_values
        new.return_values = self.return_values
        new._categoricals = self._categoricals
        new._logits = self._logits
        new._B = self._B
        new._K = self._K
        super(IndependentCategoricalDistribution, new).__init__(batch_shape, self._event_shape, validate_args=False)
        return new

    # ---------------- helpers ----------------

    def _actions_to_indices(self, actions: torch.Tensor) -> torch.Tensor:
        return actions_to_indices(actions, self.action_bins, self.bin_values)

    def _maybe_map_to_values(self, idx: torch.Tensor) -> torch.Tensor:
        # idx is [B,K] long
        if not self.return_values:
            return idx.to(dtype=torch.float32)

        if self.bin_values is None:
            raise ValueError("return_values=True requires bin_values.")

        vals = []
        for i, values in enumerate(self.bin_values):
            v = values.to(device=idx.device)
            vals.append(v[idx[..., i]])  # [B]
        return torch.stack(vals, dim=-1)  # [B,K]

    def _resolve_bin_values(self, idx: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.bin_values is not None:
            return self.bin_values[idx].to(device=device, dtype=dtype)
        bins = self.action_bins[idx]
        return torch.arange(bins, device=device, dtype=dtype)
