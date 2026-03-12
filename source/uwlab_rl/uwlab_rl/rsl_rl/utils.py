from __future__ import annotations

from typing import Any


def normalize_cfg(cfg: Any | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return dict(cfg)
