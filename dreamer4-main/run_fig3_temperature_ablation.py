#!/usr/bin/env python3
"""Run the Cu-density by temperature Fig.3 ablation diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", type=Path, default=Path("results/neurips_fig3_temperature_ablation"))
    parser.add_argument("--temperatures", type=float, nargs="+", default=[263.0, 293.0, 333.0, 373.0])
    parser.add_argument("--cu_densities", type=float, nargs="+", default=[0.005, 0.0134])
    parser.add_argument("--segments", type=int, default=50)
    parser.add_argument("--paired_samples", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("[fig3-temp] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    args = parse_args()
    cwd = Path(__file__).resolve().parent
    args.result_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    ckpts = {
        "full": "results/neurips_fixedk_matrix/diag_multik_248_seed0/best_model.pt",
        "no_duration": "results/neurips_fixedk_matrix/abl_no_duration_seed0/best_model.pt",
        "no_tau_exp": "results/neurips_fixedk_matrix/abl_no_tau_exp_seed0/best_model.pt",
        "no_projection": "results/neurips_fixedk_matrix/abl_no_proj_loss_seed0/best_model.pt",
        "no_prior": "results/neurips_fixedk_matrix/abl_no_prior_rollout_seed0/best_model.pt",
        "no_edit_context": "results/neurips_fixedk_matrix/abl_no_future_candidate_aug_seed0/best_model.pt",
    }
    closed_specs = [
        ("full", ckpts["full"], "full", ["2", "4", "8"], []),
        ("no_reachability", ckpts["full"], "no_reachability", ["2", "4", "8"], []),
        (
            "no_inventory",
            ckpts["full"],
            "no_inventory",
            ["2", "4", "8"],
            ["--raw_changed_budget_multiplier", "8.0", "--inventory_stress_mode", "vacancy_bias"],
        ),
        ("no_continuous_time", ckpts["full"], "no_continuous_time", ["2", "4", "8"], []),
    ]
    paired_specs = [
        ("full", "results/neurips_fixedk_matrix/full_seed0/best_model.pt"),
        ("no_duration", ckpts["no_duration"]),
        ("no_tau_exp", ckpts["no_tau_exp"]),
        ("no_projection", ckpts["no_projection"]),
        ("no_prior", ckpts["no_prior"]),
        ("no_edit_context", ckpts["no_edit_context"]),
    ]

    manifest: dict[str, object] = {
        "temperatures": [float(x) for x in args.temperatures],
        "cu_densities": [float(x) for x in args.cu_densities],
        "segments": int(args.segments),
        "paired_samples": int(args.paired_samples),
        "closed_loop": [],
        "paired": [],
    }
    case_items = [
        (f"Cu{str(float(cu)).replace('.', 'p')}_T{int(round(temp))}", float(cu), float(temp))
        for cu in args.cu_densities
        for temp in args.temperatures
    ]
    for case_idx, (case_code, cu_density, temp) in enumerate(case_items):
        for row_idx, (name, checkpoint, constraint_mode, segment_ks, extra_args) in enumerate(closed_specs):
            out_dir = args.result_root / "closed_loop" / case_code / name
            output = out_dir / "eval_closed_loop.json"
            manifest["closed_loop"].append(
                {
                    "case": case_code,
                    "cu_density": float(cu_density),
                    "temperature": float(temp),
                    "name": name,
                    "checkpoint": checkpoint,
                    "constraint_mode": constraint_mode,
                    "output": str(output),
                }
            )
            if output.exists() and not args.force:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            run(
                [
                    sys.executable,
                    "eval_macro_closed_loop_rollout.py",
                    "--checkpoint",
                    checkpoint,
                    "--device",
                    args.device,
                    "--seed",
                    str(case_idx * 1009 + row_idx),
                    "--rollout_segments",
                    str(args.segments),
                    "--reference_mode",
                    "on_policy_teacher_probe",
                    "--planner_segment_ks",
                    *segment_ks,
                    "--constraint_mode",
                    constraint_mode,
                    "--min_projected_changed_sites",
                    "0",
                    "--temperature_override",
                    str(float(temp)),
                    "--cu_density_override",
                    str(float(cu_density)),
                    "--max_episode_steps_override",
                    "5000",
                    "--output",
                    str(output),
                    "--print_segments",
                    "0",
                    "--progress_every",
                    "0",
                    *extra_args,
                ],
                cwd=cwd,
                env=env,
            )

    temp_codes = [f"T{int(round(temp))}" for temp in args.temperatures]
    for cu_density in args.cu_densities:
        cu_code = f"Cu{str(float(cu_density)).replace('.', 'p')}"
        paired_root = args.result_root / "paired" / cu_code
        paired_root.mkdir(parents=True, exist_ok=True)
        for name, checkpoint in paired_specs:
            output = paired_root / f"{name}.json"
            manifest["paired"].append(
                {"cu_density": float(cu_density), "name": name, "checkpoint": checkpoint, "output": str(output)}
            )
            if output.exists() and not args.force:
                continue
            run(
                [
                    sys.executable,
                    "eval_fig2_temperature_cases.py",
                    "--checkpoint",
                    checkpoint,
                    "--output",
                    str(output),
                    "--temperatures",
                    *[str(float(temp)) for temp in args.temperatures],
                    "--temperature_codes",
                    *temp_codes,
                    "--samples_per_temperature",
                    str(args.paired_samples),
                    "--long_segments",
                    "0",
                    "--device",
                    args.device,
                    "--cu_density",
                    str(float(cu_density)),
                    "--max_episode_steps_override",
                    "5000",
                    "--min_projected_changed_sites",
                    "0",
                    "--progress_every",
                    "0",
                ],
                cwd=cwd,
                env=env,
            )

    (args.result_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fig3-temp] wrote {args.result_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
