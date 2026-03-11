#!/usr/bin/env python3
"""Batch point-mass eval runner for supervised context checkpoints.

This script replaces the bash launcher with equivalent behavior:
- per-model closed-loop then open-loop eval
- optional in-file defaults for model names/context paths
- optional CLI CSV overrides
- optional GPU ID list with one active job per GPU slot
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
INFER_SCRIPT = SCRIPT_DIR / "infer_point_mass_supervised.py"

LOGS_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "supervised_context_pm"
OUTPUT_ROOT = LOGS_ROOT / "batch_eval_outputs"

# -----------------------------------------------------------------------------
# Set your default batch here so you can run without passing model/context CLI.
# - Every model runs with every context path (Cartesian product).
# -----------------------------------------------------------------------------
MODEL_NAMES: List[str] = [
    "pm_sweep_0_seed1_hd64_L6_H4_do0.2_lr1e-4_wd0.0_ictfalse",
    "pm_sweep_0_seed1_hd64_L6_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed1_hd64_L8_H4_do0.2_lr1e-4_wd0.0_ictfalse",
    "pm_sweep_0_seed1_hd64_L8_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed1_hd64_L12_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed1_hd128_L6_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed1_hd128_L8_H4_do0.2_lr1e-4_wd0.0_ictfalse",
    "pm_sweep_0_seed1_hd128_L8_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed1_hd128_L12_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed1_hd256_L6_H4_do0.2_lr1e-4_wd0.0_ictfalse",
    "pm_sweep_0_seed1_hd256_L6_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed1_hd256_L8_H4_do0.2_lr1e-4_wd0.0_ictfalse",
    "pm_sweep_0_seed1_hd256_L8_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed1_hd256_L12_H4_do0.2_lr1e-4_wd0.0_ictfalse",
    "pm_sweep_0_seed1_hd256_L12_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed2_hd64_L6_H4_do0.2_lr1e-4_wd0.0_ictfalse",
    "pm_sweep_0_seed2_hd64_L6_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed2_hd64_L8_H4_do0.2_lr1e-4_wd0.0_ictfalse",
    "pm_sweep_0_seed2_hd64_L8_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed2_hd64_L12_H4_do0.2_lr1e-4_wd0.0_ictfalse",
    "pm_sweep_0_seed2_hd64_L12_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed2_hd128_L6_H4_do0.2_lr1e-4_wd0.0_ictfalse",
    "pm_sweep_0_seed2_hd128_L6_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed2_hd128_L8_H4_do0.2_lr1e-4_wd0.0_ictfalse",
    "pm_sweep_0_seed2_hd128_L8_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed2_hd128_L12_H4_do0.2_lr1e-4_wd0.0_ictfalse",
    "pm_sweep_0_seed2_hd128_L12_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed2_hd256_L6_H4_do0.2_lr1e-4_wd0.0_ictfalse",
    "pm_sweep_0_seed2_hd256_L6_H4_do0.2_lr1e-4_wd0.0_icttrue",
    "pm_sweep_0_seed2_hd256_L8_H4_do0.2_lr1e-4_wd0.0_ictfalse",
    "pm_sweep_0_seed2_hd256_L8_H4_do0.2_lr1e-4_wd0.0_icttrue",
    # "2026-03-10_12-30-00",
]
CONTEXT_PATHS: List[str] = [
    # "/home/ubuntu/lti/dm_control/datasets/pm_lower_freq_train.pt",
    "/home/ubuntu/lti/dm_control/datasets/pm_lower_freq_val.pt",
]

# Optional GPU assignment list. If DEVICE is unset and this list is non-empty,
# jobs are assigned round-robin to cuda:<id> with one active job per GPU slot.
CUDA_DEVICE_IDS: List[str] = [
    "4",
    "5",
    "6",
    "7",
]

TASK = "easy"
NUM_EPISODES = 50
TIME_LIMIT_SEC = 5.0
MAX_STEPS: int | None = None
GROUP_INDEX = -1
NUM_CONTEXT_TRAJS = 3
SEED = 0
DEVICE = ""
PYTHON_BIN = "python"


@dataclass
class JobSpec:
    model_name: str
    context_path: Path
    checkpoint_path: Path
    run_log: Path
    closed_out: Path
    open_out: Path
    seed: int
    device: str
    gpu_slot: str | None


def _parse_csv_list(value: str | None) -> List[str]:
    if value is None or value.strip() == "":
        return []
    return [term.strip() for term in value.split(",") if term.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run closed/open-loop eval for many supervised-context checkpoints."
    )
    parser.add_argument("--checkpoint-num", type=int, required=True)
    parser.add_argument("--model-names", type=str, default=None)
    parser.add_argument("--context-paths", type=str, default=None)
    parser.add_argument("--logs-root", type=str, default=str(LOGS_ROOT))
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--task", choices=["easy", "hard"], default=TASK)
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--time-limit-sec", type=float, default=TIME_LIMIT_SEC)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--group-index", type=int, default=GROUP_INDEX)
    parser.add_argument("--num-context-trajs", type=int, default=NUM_CONTEXT_TRAJS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--cuda-device-ids", type=str, default=None)
    parser.add_argument("--python-bin", type=str, default=PYTHON_BIN)
    return parser.parse_args()


def _build_jobs(args: argparse.Namespace) -> List[JobSpec]:
    logs_root = Path(args.logs_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_names = _parse_csv_list(args.model_names) or list(MODEL_NAMES)
    context_paths_raw = _parse_csv_list(args.context_paths) or list(CONTEXT_PATHS)
    cuda_device_ids = _parse_csv_list(args.cuda_device_ids) or list(CUDA_DEVICE_IDS)

    if len(model_names) == 0:
        raise ValueError("No model names configured. Set MODEL_NAMES or pass --model-names.")
    if len(context_paths_raw) == 0:
        raise ValueError("No context paths configured. Set CONTEXT_PATHS or pass --context-paths.")

    context_paths = [Path(p).expanduser().resolve() for p in context_paths_raw]
    checkpoint_file = f"model_{args.checkpoint_num:06d}.pt"

    jobs: List[JobSpec] = []
    for model_name in model_names:
        checkpoint_path = logs_root / model_name / checkpoint_file
        if not checkpoint_path.is_file():
            print(f"Warning: checkpoint missing, skipping {model_name}: {checkpoint_path}", file=sys.stderr)
            continue

        for context_idx, context_path in enumerate(context_paths):
            if not context_path.is_file():
                print(
                    f"Warning: context file missing, skipping {model_name}: {context_path}",
                    file=sys.stderr,
                )
                continue

            assigned_device = args.device.strip()
            gpu_slot: str | None = None
            if assigned_device == "":
                if len(cuda_device_ids) > 0:
                    gpu_slot = cuda_device_ids[len(jobs) % len(cuda_device_ids)]
                    assigned_device = f"cuda:{gpu_slot}"
                else:
                    assigned_device = "cuda"

            context_tag = f"context_{context_idx:03d}_{context_path.stem}"
            per_job_dir = output_root / model_name / context_tag
            per_job_dir.mkdir(parents=True, exist_ok=True)
            jobs.append(
                JobSpec(
                    model_name=model_name,
                    context_path=context_path,
                    checkpoint_path=checkpoint_path,
                    run_log=per_job_dir / f"run_ckpt_{args.checkpoint_num}.log",
                    closed_out=per_job_dir / f"rollouts_ckpt_{args.checkpoint_num}_closed.pt",
                    open_out=per_job_dir / f"rollouts_ckpt_{args.checkpoint_num}_open.pt",
                    seed=args.seed + len(jobs),
                    device=assigned_device,
                    gpu_slot=gpu_slot,
                )
            )
    return jobs


def _job_shell_command(job: JobSpec, args: argparse.Namespace) -> str:
    common = (
        f"--checkpoint '{job.checkpoint_path}' "
        f"--context-path '{job.context_path}' "
        f"--task {args.task} "
        f"--time-limit-sec {args.time_limit_sec} "
        f"--num-episodes {args.num_episodes} "
        f"--group-index {args.group_index} "
        f"--num-context-trajs {args.num_context_trajs} "
        f"--seed {job.seed} "
        f"--device {job.device} "
    )
    if args.max_steps is not None:
        common += f"--max-steps {int(args.max_steps)} "
    return (
        f"set -euo pipefail; "
        f"echo \"[START] model={job.model_name} checkpoint={job.checkpoint_path} "
        f"context_path={job.context_path} device={job.device}\"; "
        f"{args.python_bin} '{INFER_SCRIPT}' {common} --save-rollouts-path '{job.closed_out}'; "
        f"{args.python_bin} '{INFER_SCRIPT}' {common} --open-loop --save-rollouts-path '{job.open_out}'; "
        f"echo \"[DONE] model={job.model_name}\""
    )


def _launch_job(job: JobSpec, args: argparse.Namespace) -> subprocess.Popen[str]:
    log_fp = open(job.run_log, "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["/bin/bash", "-lc", _job_shell_command(job, args)],
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_fp.close()
    print(f"Launched {job.model_name} + {job.context_path.name} (pid={proc.pid}), log: {job.run_log}")
    return proc


def main() -> int:
    args = parse_args()
    try:
        jobs = _build_jobs(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if len(jobs) == 0:
        print("No jobs launched.")
        return 1

    # Queue mode: one active process per GPU slot when using --cuda-device-ids
    # and no global --device override.
    queue_mode = (args.device.strip() == "") and (
        len(_parse_csv_list(args.cuda_device_ids) or list(CUDA_DEVICE_IDS)) > 0
    )

    failed = False
    progress = tqdm(total=len(jobs), desc="Completed eval jobs", unit="job")
    if queue_mode:
        slot_procs: Dict[str, subprocess.Popen[str]] = {}
        slot_names: Dict[str, str] = {}
        for job in jobs:
            assert job.gpu_slot is not None
            slot = job.gpu_slot
            if slot in slot_procs:
                prior_proc = slot_procs[slot]
                prior_name = slot_names[slot]
                if prior_proc.wait() == 0:
                    print(f"[OK] {prior_name}")
                else:
                    print(f"[FAIL] {prior_name}", file=sys.stderr)
                    failed = True
                progress.update(1)
            slot_procs[slot] = _launch_job(job, args)
            slot_names[slot] = f"{job.model_name} + {job.context_path.name}"

        print("Waiting for final job on each GPU slot...")
        for slot, proc in slot_procs.items():
            name = slot_names[slot]
            if proc.wait() == 0:
                print(f"[OK] {name}")
            else:
                print(f"[FAIL] {name}", file=sys.stderr)
                failed = True
            progress.update(1)
    else:
        procs: List[subprocess.Popen[str]] = []
        names: List[str] = []
        for job in jobs:
            procs.append(_launch_job(job, args))
            names.append(f"{job.model_name} + {job.context_path.name}")

        print(f"Waiting for {len(procs)} eval job(s)...")
        for proc, name in zip(procs, names):
            if proc.wait() == 0:
                print(f"[OK] {name}")
            else:
                print(f"[FAIL] {name}", file=sys.stderr)
                failed = True
            progress.update(1)

    progress.close()

    if failed:
        print(
            f"Completed with failures. Check per-model logs in: {Path(args.output_root).expanduser().resolve()}",
            file=sys.stderr,
        )
        return 1
    print("All model eval jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

