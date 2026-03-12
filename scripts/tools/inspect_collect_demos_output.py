#!/usr/bin/env python
# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect EpisodeStorage output from collect_demos.py."""

from __future__ import annotations

import argparse
from typing import Any

import torch


def _summarize_tensor(value: torch.Tensor) -> str:
    return f"shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"


def _summarize_state_dict(state: Any) -> Any:
    if isinstance(state, torch.Tensor):
        return _summarize_tensor(state)
    if not isinstance(state, dict):
        return type(state).__name__
    summary: dict[str, Any] = {}
    for asset_type, assets in state.items():
        if isinstance(assets, torch.Tensor):
            summary[asset_type] = _summarize_tensor(assets)
            continue
        if not isinstance(assets, dict):
            summary[asset_type] = type(assets).__name__
            continue
        summary[asset_type] = {}
        for asset_name, asset_data in assets.items():
            if isinstance(asset_data, torch.Tensor):
                summary[asset_type][asset_name] = _summarize_tensor(asset_data)
                continue
            if not isinstance(asset_data, dict):
                summary[asset_type][asset_name] = type(asset_data).__name__
                continue
            summary[asset_type][asset_name] = {}
            for key, value in asset_data.items():
                if isinstance(value, torch.Tensor):
                    summary[asset_type][asset_name][key] = _summarize_tensor(value)
                else:
                    summary[asset_type][asset_name][key] = type(value).__name__
    return summary


def _summarize_raw_states(raw_states: list[dict[str, Any]], max_samples: int) -> dict[str, Any]:
    if not raw_states:
        return {"count": 0}
    samples = []
    for entry in raw_states[: max_samples]:
        timestep = entry.get("timestep")
        state = entry.get("state", {})
        samples.append(
            {
                "timestep": timestep,
                "state_summary": _summarize_state_dict(state) if isinstance(state, dict) else type(state).__name__,
            }
        )
    return {"count": len(raw_states), "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect collect_demos EpisodeStorage outputs.")
    parser.add_argument("episode_file", type=str, help="Path to episodes_*.pt file.")
    parser.add_argument("--episode_index", type=int, default=0, help="Episode index to inspect.")
    parser.add_argument("--raw_state_samples", type=int, default=2, help="Number of raw state samples to show.")
    args = parser.parse_args()

    data = torch.load(args.episode_file, map_location="cpu")
    episodes = data.get("episodes", [])
    if not episodes:
        raise ValueError(f"No episodes found in {args.episode_file}")

    if args.episode_index < 0 or args.episode_index >= len(episodes):
        raise IndexError(f"episode_index out of range (0..{len(episodes) - 1})")

    episode = episodes[args.episode_index]
    print(f"Episode count: {len(episodes)}")
    print(f"Inspecting episode index: {args.episode_index}")
    print(f"Keys: {list(episode.keys())}")
    print(f"Length: {episode.get('length')}")
    print(f"Env id: {episode.get('env_id')}")

    obs = episode.get("obs")
    if isinstance(obs, dict):
        print("Obs keys:", list(obs.keys()))
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                print(f"  obs[{key}]: {_summarize_tensor(value)}")
    elif isinstance(obs, torch.Tensor):
        print(f"Obs: {_summarize_tensor(obs)}")

    actions = episode.get("actions")
    if isinstance(actions, torch.Tensor):
        print(f"Actions: {_summarize_tensor(actions)}")

    rewards = episode.get("rewards")
    if isinstance(rewards, torch.Tensor):
        print(f"Rewards: {_summarize_tensor(rewards)}")

    states = episode.get("states")
    if isinstance(states, dict):
        print("States summary:")
        print(_summarize_state_dict(states))

    physics = episode.get("physics")
    if isinstance(physics, dict):
        print("Physics keys:", list(physics.keys()))

    raw_states = episode.get("raw_states")
    if isinstance(raw_states, list):
        print("Raw states summary:")
        summary = _summarize_raw_states(raw_states, args.raw_state_samples)
        print(f"  count: {summary.get('count')}")
        for idx, sample in enumerate(summary.get("samples", [])):
            print(f"\n  sample {idx}:")
            print(f"    timestep: {sample.get('timestep')}")
            state_summary = sample.get("state_summary")
            if isinstance(state_summary, dict):
                print("    state_summary:")
                for asset_type, assets in state_summary.items():
                    print(f"      {asset_type}: {assets}")
            else:
                print(f"    state_summary: {state_summary}")
    else:
        print("Raw states: not present or not a list.")


if __name__ == "__main__":
    main()
