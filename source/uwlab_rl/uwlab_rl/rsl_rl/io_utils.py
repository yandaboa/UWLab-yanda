from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch
import yaml


def class_to_dict(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, Mapping):
        return {key: class_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(class_to_dict(value) for value in obj)
    if hasattr(obj, "__dict__"):
        return {
            key: class_to_dict(value)
            for key, value in obj.__dict__.items()
            if not key.startswith("__") and not callable(value)
        }
    return obj


def dump_yaml(filename: str, data: dict[str, Any] | object, sort_keys: bool = False) -> None:
    if not filename.endswith("yaml"):
        filename += ".yaml"
    output_dir = os.path.dirname(filename)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)
    payload = data if isinstance(data, dict) else class_to_dict(data)
    with open(filename, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, default_flow_style=False, sort_keys=sort_keys)
