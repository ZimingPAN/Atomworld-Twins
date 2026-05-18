#!/usr/bin/env python3
"""Benchmark online multi-k world-model planner decision latency."""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import eval_macro_long_trajectory as long_eval
import train_dreamer_macro_edit as mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--duration_checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup_segments", type=int, default=10)
    parser.add_argument("--timed_segments", type=int, default=200)
    parser.add_argument("--max_episode_steps_override", type=int, default=None)
    parser.add_argument("--lattice_shape", type=str, default=None,
                        help="Optional lattice shape override such as 40x40x40.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Optional physical temperature override for timing states.")
    parser.add_argument("--cu_density", type=float, default=None,
                        help="Optional Cu density override for timing states.")
    parser.add_argument("--planner_segment_ks", type=int, nargs="+", default=None)
    parser.add_argument("--min_projected_changed_sites", type=int, default=2)
    parser.add_argument("--max_candidate_sites_override", type=int, default=None)
    parser.add_argument("--duration_source", type=str, default="model", choices=["model", "baseline", "blend"])
    parser.add_argument("--duration_blend_alpha", type=float, default=1.0)
    parser.add_argument("--duration_log_offset", type=float, default=0.0)
    parser.add_argument("--planner_tau_source", type=str, default=None, choices=["model", "baseline", "blend"])
    parser.add_argument("--planner_tau_blend_alpha", type=float, default=None)
    parser.add_argument("--planner_score_mode", type=str, default="energy_per_tau",
                        choices=["energy_per_tau", "energy_per_sqrt_tau", "energy"])
    parser.add_argument("--planner_tau_residual_penalty", type=float, default=0.0)
    parser.add_argument("--planner_k_penalty_power", type=float, default=0.0)
    parser.add_argument("--planner_duration_checkpoint_source", type=str, default="duration",
                        choices=["primary", "duration"])
    parser.add_argument("--amp_dtype", choices=["none", "float16", "bfloat16"], default="none")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--allow_teacher_noop_segments", action="store_true")
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_lattice_shape(token: str | None) -> tuple[int, int, int] | None:
    if token is None:
        return None
    parts = token.lower().replace(",", "x").replace("*", "x").split("x")
    if len(parts) != 3:
        raise ValueError(f"Expected lattice shape like 40x40x40, got {token!r}")
    shape = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"Lattice dimensions must be positive, got {token!r}")
    return shape


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def autocast_context(device: str, amp_dtype: str):
    if amp_dtype == "none" or not device.startswith("cuda") or not torch.cuda.is_available():
        return contextlib.nullcontext()
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def summarize(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "p80": 0.0,
            "p95": 0.0,
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "p80": float(np.percentile(arr, 80)),
        "p95": float(np.percentile(arr, 95)),
    }


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)
    planner_tau_source = args.planner_tau_source or args.duration_source
    planner_tau_blend_alpha = (
        float(args.duration_blend_alpha)
        if args.planner_tau_blend_alpha is None
        else float(args.planner_tau_blend_alpha)
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    ckpt_args = ckpt["args"]
    reward_scale = float(ckpt_args.get("reward_scale", 1.0))
    reward_prediction_source = str(ckpt_args.get("reward_prediction_source", "raw"))
    ckpt_segment_ks = long_eval._segment_ks_from_ckpt_args(ckpt_args)
    if args.planner_segment_ks:
        horizon_choices = sorted({int(k) for k in args.planner_segment_ks})
    elif len(ckpt_segment_ks) > 1:
        horizon_choices = ckpt_segment_ks
    else:
        horizon_choices = [int(ckpt_args["segment_k"])]
    planner_enabled = len(horizon_choices) > 1
    max_seed_vacancies = int(ckpt_args["max_seed_vacancies"])
    max_candidate_sites = int(
        args.max_candidate_sites_override
        if args.max_candidate_sites_override is not None
        else ckpt_args["max_candidate_sites"]
    )

    model = long_eval._build_model(ckpt, args.device)
    duration_model = None
    if args.duration_checkpoint is not None:
        duration_ckpt = torch.load(args.duration_checkpoint, map_location=args.device, weights_only=False)
        duration_model = long_eval._build_model(duration_ckpt, args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env_cfg = long_eval._build_env_cfg(ckpt_args, max_episode_steps_override=args.max_episode_steps_override)
    lattice_shape_override = parse_lattice_shape(args.lattice_shape)
    if lattice_shape_override is not None:
        env_cfg["lattice_size"] = lattice_shape_override
    if args.temperature is not None:
        env_cfg["temperature"] = float(args.temperature)
    if args.cu_density is not None:
        env_cfg["cu_density"] = float(args.cu_density)
    env = mod.MacroKMCEnv(env_cfg)
    env.reset()

    decision_times: list[float] = []
    per_horizon_times: dict[int, list[float]] = {int(k): [] for k in horizon_choices}
    selected_ks: list[int] = []
    projected_changed_counts: list[float] = []
    stop_reason = "completed"
    stop_segment: dict[str, object] | None = None
    total_segments = max(int(args.warmup_segments), 0) + max(int(args.timed_segments), 0)

    with torch.inference_mode():
        for segment_idx in range(total_segments):
            candidates = []
            horizon_elapsed: dict[int, float] = {}
            synchronize(args.device)
            t0 = time.perf_counter()
            with autocast_context(args.device, args.amp_dtype):
                for item_k in horizon_choices:
                    synchronize(args.device)
                    hk_t0 = time.perf_counter()
                    item = long_eval._predict_candidate_for_horizon(
                        model=model,
                        duration_model=duration_model,
                        env=env,
                        horizon_k=int(item_k),
                        max_seed_vacancies=max_seed_vacancies,
                        max_candidate_sites=max_candidate_sites,
                        reward_scale=reward_scale,
                        device=args.device,
                        duration_source=args.duration_source,
                        planner_tau_source=planner_tau_source,
                        planner_score_mode=args.planner_score_mode,
                        planner_tau_residual_penalty=args.planner_tau_residual_penalty,
                        planner_k_penalty_power=args.planner_k_penalty_power,
                        duration_blend_alpha=args.duration_blend_alpha,
                        planner_tau_blend_alpha=planner_tau_blend_alpha,
                        duration_log_offset=args.duration_log_offset,
                        planner_tau_log_offset=args.duration_log_offset,
                        planner_duration_checkpoint_source=args.planner_duration_checkpoint_source,
                        reward_prediction_source=reward_prediction_source,
                    )
                    synchronize(args.device)
                    horizon_elapsed[int(item_k)] = time.perf_counter() - hk_t0
                    if item is not None:
                        candidates.append(item)
                effective_min_projected_changed_sites = (
                    int(args.min_projected_changed_sites) if planner_enabled else 0
                )
                selected = long_eval._choose_planner_candidate(
                    candidates,
                    min_projected_changed_sites=effective_min_projected_changed_sites,
                )
            synchronize(args.device)
            elapsed = time.perf_counter() - t0

            if segment_idx >= int(args.warmup_segments):
                decision_times.append(float(elapsed))
                for item_k, item_elapsed in horizon_elapsed.items():
                    per_horizon_times[int(item_k)].append(float(item_elapsed))

            if selected is None:
                stop_reason = "no_legal_planner_candidate"
                stop_segment = {"index": segment_idx, "planner_candidates": candidates}
                break

            selected_k = int(selected["segment_k"])
            if segment_idx >= int(args.warmup_segments):
                selected_ks.append(selected_k)
                projected_changed_counts.append(float(selected.get("projected_changed_count", 0.0)))

            teacher_segment = long_eval._collect_teacher_segment(env, horizon_k=selected_k, rng=rng)
            if teacher_segment is None:
                stop_reason = "teacher_terminal_or_action_missing"
                stop_segment = {
                    "index": segment_idx,
                    "selected_k": selected_k,
                    "planner_candidates": candidates,
                    "selected": selected,
                }
                break
            if bool(teacher_segment.get("is_noop", False)) and not args.allow_teacher_noop_segments:
                stop_reason = "noop_teacher_segment"
                stop_segment = {
                    "index": segment_idx,
                    "selected_k": selected_k,
                    "planner_candidates": candidates,
                    "selected": selected,
                    "traditional_changed_site_count": int(teacher_segment.get("changed_site_count", 0)),
                }
                break

            if args.progress_every > 0 and (segment_idx + 1) % int(args.progress_every) == 0:
                timed = max(0, segment_idx + 1 - int(args.warmup_segments))
                print(
                    f"segment={segment_idx + 1}/{total_segments} timed={timed} "
                    f"last_decision_ms={elapsed * 1e3:.3f} selected_k={selected_k}",
                    flush=True,
                )

    selected_hist = {str(k): int(selected_ks.count(int(k))) for k in horizon_choices}
    output = {
        "mode": "online_multik_planner_decision_latency",
        "checkpoint": str(args.checkpoint),
        "duration_checkpoint": str(args.duration_checkpoint) if args.duration_checkpoint else None,
        "device": args.device,
        "amp_dtype": args.amp_dtype,
        "matmul_precision": args.matmul_precision,
        "seed": int(args.seed),
        "warmup_segments": int(args.warmup_segments),
        "requested_timed_segments": int(args.timed_segments),
        "timed_segments": int(len(decision_times)),
        "stop_reason": stop_reason,
        "stop_segment": stop_segment,
        "segment_ks": [int(k) for k in horizon_choices],
        "planner_enabled": bool(planner_enabled),
        "min_projected_changed_sites": int(args.min_projected_changed_sites),
        "max_seed_vacancies": int(max_seed_vacancies),
        "max_candidate_sites": int(max_candidate_sites),
        "checkpoint_max_candidate_sites": int(ckpt_args["max_candidate_sites"]),
        "env_cfg": {
            "lattice_size": [int(x) for x in env_cfg["lattice_size"]],
            "temperature": float(env_cfg["temperature"]),
            "cu_density": float(env_cfg["cu_density"]),
            "v_density": float(env_cfg.get("v_density", 0.0)),
        },
        "duration_source": args.duration_source,
        "planner_tau_source": planner_tau_source,
        "planner_score_mode": args.planner_score_mode,
        "reward_prediction_source": reward_prediction_source,
        "decision_wall_s": summarize(decision_times),
        "decision_wall_ms": {key: (float(value) * 1e3 if isinstance(value, float) else value)
                             for key, value in summarize(decision_times).items()},
        "per_horizon_wall_s": {str(k): summarize(v) for k, v in per_horizon_times.items()},
        "selected_k_histogram": selected_hist,
        "projected_changed_count": summarize(projected_changed_counts),
        "values_s": [float(x) for x in decision_times],
    }
    output["decisions_per_wall_second"] = float(len(decision_times) / max(float(np.sum(decision_times)), 1e-12))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ["timed_segments", "stop_reason", "decision_wall_ms", "decisions_per_wall_second", "selected_k_histogram"]}, indent=2))


if __name__ == "__main__":
    main()
