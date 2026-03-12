from __future__ import annotations

from typing import Sequence

import torch


def actions_to_indices(
    actions: torch.Tensor,
    action_bins: Sequence[int],
    bin_values: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Map actions [B,K] to per-dim bin indices, using nearest bin values when available."""
    if actions.dtype == torch.long:
        idx = actions
    else:
        if bin_values is not None:
            assert len(bin_values) == len(action_bins), "bin_values must match action_bins length."
            idxs = []
            for dim, values in enumerate(bin_values):
                v = values.to(device=actions.device, dtype=actions.dtype).view(1, -1)
                a = actions[..., dim].unsqueeze(-1)
                idxs.append((a - v).abs().argmin(dim=-1))
            idx = torch.stack(idxs, dim=-1)
        else:
            idx = actions.round().to(torch.long)
    idx_clamped = [idx[..., dim].clamp(0, int(bins) - 1) for dim, bins in enumerate(action_bins)]
    return torch.stack(idx_clamped, dim=-1)


def indices_to_actions(
    indices: torch.Tensor,
    action_bins: Sequence[int],
    bin_values: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Map per-dim bin indices back to action values using bin_values."""
    if indices.dtype != torch.long:
        indices = indices.to(dtype=torch.long)
    if bin_values is None:
        raise AssertionError("indices_to_actions requires bin_values for discretized actions.")
    assert len(bin_values) == len(action_bins), "bin_values must match action_bins length."
    actions = []
    for dim, bins in enumerate(action_bins):
        values = bin_values[dim].to(device=indices.device)
        idx = indices[..., dim].clamp(0, int(bins) - 1)
        actions.append(values[idx])
    return torch.stack(actions, dim=-1)
