from __future__ import annotations

from dataclasses import asdict
from typing import Callable, cast

import torch
from omegaconf import OmegaConf

from uwlab_rl.rsl_rl.supervised_context_cfg import SupervisedContextTrainerCfg


def load_cfg_dict(cfg_path: str | None) -> dict | None:
    if cfg_path is None:
        return None
    if cfg_path.endswith(".pt"):
        payload = torch.load(cfg_path, map_location="cpu")
        if isinstance(payload, dict) and "cfg" in payload:
            payload = payload["cfg"]
        if not isinstance(payload, dict):
            raise ValueError("Config checkpoint must contain a dict or a dict under 'cfg'.")
        return payload
    if cfg_path.endswith((".yaml", ".yml")):
        container = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        if not isinstance(container, dict):
            raise ValueError("YAML config must contain a dictionary at the top level.")
        return container
    raise ValueError("Unsupported config file type. Use .pt, .yaml, or .yml.")


def apply_cfg_overrides(
    cfg: SupervisedContextTrainerCfg,
    cfg_dict: dict | None,
    overrides: list[str],
) -> None:
    merged = OmegaConf.create(asdict(cfg))
    if cfg_dict:
        merged = OmegaConf.merge(merged, OmegaConf.create(cfg_dict))
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    merged_dict = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(merged_dict, dict):
        raise ValueError("Merged config must resolve to a dictionary.")
    from_dict_fn = cast(Callable[[dict], None], getattr(cfg, "from_dict"))
    from_dict_fn(merged_dict)
