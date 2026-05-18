#!/usr/bin/env python3
"""Benchmark Fig.4 runtime cases over Cu concentration, temperature, and lattice size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import eval_macro_long_trajectory as long_eval
from benchmark_lattice_size_speedup import (
    collect_macro_end_to_end_runtime,
    collect_macro_runtime,
    collect_teacher_runtime,
    shape_product,
)


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cu_densities", type=float, nargs="+", default=[0.005, 0.0134])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[663.0, 693.0, 733.0, 773.0])
    parser.add_argument("--lattice_edges", type=int, nargs="+", default=[20, 30, 40, 50, 60])
    parser.add_argument("--segment_k", type=int, default=8)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--teacher_repeats", type=int, default=2)
    parser.add_argument("--teacher_warmup_segments", type=int, default=5)
    parser.add_argument("--macro_repeats", type=int, default=30)
    parser.add_argument("--macro_warmup", type=int, default=10)
    parser.add_argument(
        "--macro_timing_scope",
        choices=["inference", "end_to_end"],
        default="inference",
        help=(
            "inference times batched model forward on prebuilt active-region tensors; "
            "end_to_end times candidate/tensor construction, model forward, and projection."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--candidate_limit", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp_dtype", choices=["none", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--max_episode_steps_override", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def concentration_code(index: int) -> str:
    if not 0 <= index < 26:
        raise ValueError(f"Too many concentration levels for A-Z codes: {index + 1}")
    return chr(ord("A") + index)


def lattice_code(index: int) -> str:
    if not 0 <= index < len(ROMAN):
        raise ValueError(f"Too many lattice levels for built-in roman codes: {index + 1}")
    return ROMAN[index]


def build_output(
    args: argparse.Namespace,
    *,
    cu_levels: list[float],
    temp_levels: list[float],
    lattice_edges: list[int],
    rows: list[dict],
    complete: bool,
) -> dict:
    total_cases = len(cu_levels) * len(temp_levels) * len(lattice_edges)
    return {
        "mode": "fig4_cu_temperature_lattice_runtime_case_grid",
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "amp_dtype": args.amp_dtype,
        "matmul_precision": args.matmul_precision,
        "seed": int(args.seed),
        "cu_densities": cu_levels,
        "temperatures": temp_levels,
        "lattice_edges": lattice_edges,
        "samples": int(args.samples),
        "segment_k": int(args.segment_k),
        "teacher_repeats": int(args.teacher_repeats),
        "teacher_warmup_segments": int(args.teacher_warmup_segments),
        "macro_repeats": int(args.macro_repeats),
        "macro_warmup": int(args.macro_warmup),
        "macro_timing_scope": args.macro_timing_scope,
        "batch_size": int(args.batch_size),
        "candidate_limit": int(args.candidate_limit),
        "complete": bool(complete),
        "completed_cases": len(rows),
        "total_cases": total_cases,
        "codebook": {
            "cu_concentration": {
                concentration_code(idx): float(value)
                for idx, value in enumerate(cu_levels)
            },
            "temperature": {
                str(idx + 1): float(value)
                for idx, value in enumerate(temp_levels)
            },
            "cubic_lattice_size": {
                lattice_code(idx): f"{int(edge)}^3"
                for idx, edge in enumerate(lattice_edges)
            },
        },
        "runtime_scope": {
            "akmc": "teacher KMC replay for the same number of fixed-k macro segments; environment construction is excluded",
            "atomworld": (
                "batched world-model forward inference on prebuilt current-state reachable active-region tensors; candidate construction is excluded"
                if args.macro_timing_scope == "inference"
                else "per-state macro-prediction path including reachable candidate construction, tensor creation, world-model forward inference, and inventory projection; teacher state advance used only to sample benchmark states is excluded"
            ),
        },
        "rows": rows,
    }


def write_output(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    ckpt_args = ckpt["args"]
    model = long_eval._build_model(ckpt, args.device)

    cu_levels = [float(x) for x in args.cu_densities]
    temp_levels = [float(x) for x in args.temperatures]
    lattice_edges = [int(x) for x in args.lattice_edges]
    rows = []

    for cu_idx, cu_density in enumerate(cu_levels):
        cu_code = concentration_code(cu_idx)
        for temp_idx, temperature in enumerate(temp_levels):
            temp_code = str(temp_idx + 1)
            for lattice_idx, edge in enumerate(lattice_edges):
                lat_code = lattice_code(lattice_idx)
                shape = (int(edge), int(edge), int(edge))
                case_code = f"{cu_code}{temp_code}{lat_code}"
                row_index = len(rows)
                row_seed = int(args.seed) + row_index * 1000
                print(
                    f"[fig4-case] {case_code} cu={cu_density:g} T={temperature:g}K "
                    f"shape={edge}^3 start",
                    flush=True,
                )
                teacher, micro_events = collect_teacher_runtime(
                    ckpt_args=ckpt_args,
                    shape=shape,
                    segment_k=int(args.segment_k),
                    samples=int(args.samples),
                    repeats=int(args.teacher_repeats),
                    warmup_segments=int(args.teacher_warmup_segments),
                    max_episode_steps=int(args.max_episode_steps_override),
                    base_seed=row_seed,
                    temperature=temperature,
                    cu_density=cu_density,
                )
                if args.macro_timing_scope == "end_to_end":
                    macro = collect_macro_end_to_end_runtime(
                        model=model,
                        ckpt_args=ckpt_args,
                        shape=shape,
                        segment_k=int(args.segment_k),
                        samples=int(args.samples),
                        candidate_limit=int(args.candidate_limit),
                        repeats=int(args.macro_repeats),
                        warmup=int(args.macro_warmup),
                        device=args.device,
                        amp_dtype=args.amp_dtype,
                        max_episode_steps=int(args.max_episode_steps_override),
                        base_seed=row_seed + 97,
                        temperature=temperature,
                        cu_density=cu_density,
                    )
                else:
                    macro = collect_macro_runtime(
                        model=model,
                        ckpt_args=ckpt_args,
                        shape=shape,
                        segment_k=int(args.segment_k),
                        samples=int(args.samples),
                        candidate_limit=int(args.candidate_limit),
                        batch_size=int(args.batch_size),
                        repeats=int(args.macro_repeats),
                        warmup=int(args.macro_warmup),
                        device=args.device,
                        amp_dtype=args.amp_dtype,
                        max_episode_steps=int(args.max_episode_steps_override),
                        base_seed=row_seed + 97,
                        temperature=temperature,
                        cu_density=cu_density,
                    )
                speedup = float(float(teacher["mean"]) / max(float(macro["mean"]), 1e-12))
                row = {
                    "case_code": case_code,
                    "cu_code": cu_code,
                    "temperature_code": temp_code,
                    "lattice_code": lat_code,
                    "cu_density": float(cu_density),
                    "temperature": float(temperature),
                    "lattice_edge": int(edge),
                    "lattice_shape": [int(edge), int(edge), int(edge)],
                    "lattice_size_l3": int(shape_product(shape)),
                    "bcc_sites": 2 * int(shape_product(shape)),
                    "samples": int(args.samples),
                    "segment_k": int(args.segment_k),
                    "teacher_represented_micro_events": int(micro_events),
                    "akmc_runtime_s": teacher,
                    "atomworld_runtime_s": macro,
                    "speedup_akmc_over_atomworld": speedup,
                }
                rows.append(row)
                write_output(
                    args.output,
                    build_output(
                        args,
                        cu_levels=cu_levels,
                        temp_levels=temp_levels,
                        lattice_edges=lattice_edges,
                        rows=rows,
                        complete=False,
                    ),
                )
                print(
                    json.dumps(
                        {
                            "case_code": case_code,
                            "akmc_runtime_s": float(teacher["mean"]),
                            "atomworld_runtime_s": float(macro["mean"]),
                            "speedup": speedup,
                        },
                        indent=2,
                    ),
                    flush=True,
                )

    output = build_output(
        args,
        cu_levels=cu_levels,
        temp_levels=temp_levels,
        lattice_edges=lattice_edges,
        rows=rows,
        complete=True,
    )
    write_output(args.output, output)
    print(f"[fig4-case] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
