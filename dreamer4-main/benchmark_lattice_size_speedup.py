#!/usr/bin/env python3
"""Benchmark AtomWorld-Mirror runtime scaling over cubic lattice sizes."""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

import eval_macro_long_trajectory as long_eval
import train_dreamer_macro_edit as mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lattice_edges", type=int, nargs="+", default=[20, 30, 40, 50, 60])
    parser.add_argument("--lattice_shapes", type=str, nargs="+", default=None,
                        help="Optional explicit shapes such as 8x10x5. Overrides --lattice_edges.")
    parser.add_argument("--segment_k", type=int, default=8)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--teacher_repeats", type=int, default=3)
    parser.add_argument("--teacher_warmup_segments", type=int, default=5)
    parser.add_argument("--macro_repeats", type=int, default=50)
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
    parser.add_argument("--temperature", type=float, default=None,
                        help="Optional physical temperature override for all benchmark states.")
    parser.add_argument("--cu_density", type=float, default=None,
                        help="Optional Cu density override for all benchmark states.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp_dtype", choices=["none", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--max_episode_steps_override", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_shape(token: str) -> tuple[int, int, int]:
    parts = token.lower().replace(",", "x").replace("*", "x").split("x")
    if len(parts) != 3:
        raise ValueError(f"Expected lattice shape like 20x20x16, got {token!r}")
    shape = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"Lattice shape dimensions must be positive, got {token!r}")
    return shape


def shape_product(shape: Iterable[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return int(out)


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def autocast_context(device: str, amp_dtype: str):
    if amp_dtype == "none" or not device.startswith("cuda") or not torch.cuda.is_available():
        return contextlib.nullcontext()
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def summarize(values: list[float]) -> dict[str, float | int | list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "values": []}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "values": [float(x) for x in arr.tolist()],
    }


def build_env_cfg(
    ckpt_args: dict[str, object],
    shape: tuple[int, int, int],
    max_episode_steps: int,
    *,
    temperature: float | None = None,
    cu_density: float | None = None,
) -> dict[str, object]:
    cfg = long_eval._build_env_cfg(ckpt_args, max_episode_steps_override=max_episode_steps)
    cfg["lattice_size"] = tuple(int(dim) for dim in shape)
    if temperature is not None:
        cfg["temperature"] = float(temperature)
    if cu_density is not None:
        cfg["cu_density"] = float(cu_density)
    return cfg


def collect_teacher_runtime(
    *,
    ckpt_args: dict[str, object],
    shape: tuple[int, int, int],
    segment_k: int,
    samples: int,
    repeats: int,
    warmup_segments: int,
    max_episode_steps: int,
    base_seed: int,
    temperature: float | None = None,
    cu_density: float | None = None,
) -> tuple[dict[str, float | int | list[float]], int]:
    run_times: list[float] = []
    completed_counts: list[int] = []
    cfg = build_env_cfg(
        ckpt_args,
        shape=shape,
        max_episode_steps=max_episode_steps,
        temperature=temperature,
        cu_density=cu_density,
    )
    for repeat_idx in range(max(int(repeats), 1)):
        np.random.seed(base_seed + repeat_idx)
        torch.manual_seed(base_seed + repeat_idx)
        rng = np.random.default_rng(base_seed + repeat_idx)
        env = mod.MacroKMCEnv(cfg)
        env.reset()
        for _ in range(max(int(warmup_segments), 0)):
            segment = long_eval._collect_teacher_segment(env, horizon_k=int(segment_k), rng=rng)
            if segment is None:
                env = mod.MacroKMCEnv(cfg)
                env.reset()
        completed = 0
        t0 = time.perf_counter()
        while completed < int(samples):
            segment = long_eval._collect_teacher_segment(env, horizon_k=int(segment_k), rng=rng)
            if segment is None:
                env = mod.MacroKMCEnv(cfg)
                env.reset()
                continue
            completed += 1
        run_times.append(time.perf_counter() - t0)
        completed_counts.append(completed)
    summary = summarize(run_times)
    summary["segments"] = int(samples)
    summary["segment_k"] = int(segment_k)
    summary["segments_per_wall_second"] = float(int(samples) / max(float(summary["mean"]), 1e-12))
    summary["micro_events_per_wall_second"] = float((int(samples) * int(segment_k)) / max(float(summary["mean"]), 1e-12))
    summary["completed_counts"] = [int(x) for x in completed_counts]
    return summary, int(samples) * int(segment_k)


def forward_batch(
    model: mod.MacroDreamerEditModel,
    tensors: dict[str, torch.Tensor],
    *,
    include_projection: bool = False,
) -> None:
    global_latent = model.encode_global(tensors["start_obs"])
    site_latent, patch_latent = model.encode_patch(
        positions=tensors["candidate_positions"],
        nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
        reach_depth=tensors["reach_depth"],
        is_start_vacancy=tensors["is_start_vacancy"],
        type_ids=tensors["current_types"],
        node_mask=tensors["candidate_mask"],
        global_summary=tensors["global_summary"],
        box_dims=tensors["box_dims"],
    )
    prior_mu, prior_logvar = model.prior_stats(global_latent, tensors["global_summary"], tensors["horizon_k"])
    path_latent = model.sample_path_latent(prior_mu, prior_logvar, deterministic=True)
    next_pred = model.predict_next_global(global_latent, path_latent, tensors["horizon_k"])
    change_logits, raw_type_logits = model.decode_edit(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=next_pred,
        path_latent=path_latent,
        horizon_k=tensors["horizon_k"],
        current_types=tensors["current_types"],
    )
    duration_outputs = mod._predict_reward_and_duration_outputs(
        model,
        global_latent,
        next_pred,
        path_latent,
        tensors["global_summary"],
        tensors["horizon_k"],
        patch_latent=patch_latent,
        change_logits=change_logits,
        type_logits=raw_type_logits,
        current_types=tensors["current_types"],
        candidate_mask=tensors["candidate_mask"],
    )
    if not include_projection:
        return

    projected_types, _, _, _ = mod.project_types_by_inventory(
        current_types=tensors["current_types"],
        change_logits=change_logits,
        type_logits=raw_type_logits,
        node_mask=tensors["candidate_mask"],
        positions=tensors["candidate_positions"],
        box_dims=tensors["box_dims"],
        horizon_k=tensors["horizon_k"],
        max_changed_sites=2 * tensors["horizon_k"],
    )
    _, projected_patch_latent = model.encode_patch(
        positions=tensors["candidate_positions"],
        nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
        reach_depth=tensors["reach_depth"],
        is_start_vacancy=tensors["is_start_vacancy"],
        type_ids=projected_types,
        node_mask=tensors["candidate_mask"],
        global_summary=tensors["global_summary"],
        box_dims=tensors["box_dims"],
    )
    projected_change_logits, projected_type_logits = mod.projected_edit_logits_from_types(
        current_types=tensors["current_types"],
        projected_types=projected_types,
        candidate_mask=tensors["candidate_mask"],
    )
    if duration_outputs:
        mod._predict_reward_and_duration_outputs(
            model,
            global_latent,
            next_pred,
            path_latent,
            tensors["global_summary"],
            tensors["horizon_k"],
            patch_latent=projected_patch_latent,
            change_logits=projected_change_logits,
            type_logits=projected_type_logits,
            current_types=tensors["current_types"],
            candidate_mask=tensors["candidate_mask"],
        )


def build_macro_batches(
    *,
    model: mod.MacroDreamerEditModel,
    ckpt_args: dict[str, object],
    shape: tuple[int, int, int],
    segment_k: int,
    samples: int,
    candidate_limit: int,
    batch_size: int,
    device: str,
    max_episode_steps: int,
    base_seed: int,
    temperature: float | None = None,
    cu_density: float | None = None,
) -> list[dict[str, torch.Tensor]]:
    del model
    cfg = build_env_cfg(
        ckpt_args,
        shape=shape,
        max_episode_steps=max_episode_steps,
        temperature=temperature,
        cu_density=cu_density,
    )
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    rng = np.random.default_rng(base_seed)
    env = mod.MacroKMCEnv(cfg)
    env.reset()
    max_seed_vacancies = int(ckpt_args["max_seed_vacancies"])
    rows: list[dict[str, torch.Tensor]] = []
    attempts = 0
    while len(rows) < int(samples):
        tensors = long_eval._build_inference_tensors(
            env=env,
            max_seed_vacancies=max_seed_vacancies,
            max_candidate_sites=int(candidate_limit),
            horizon_k=int(segment_k),
            device=device,
        )
        if tensors is not None:
            rows.append(tensors)
        segment = long_eval._collect_teacher_segment(env, horizon_k=int(segment_k), rng=rng)
        if segment is None:
            env = mod.MacroKMCEnv(cfg)
            env.reset()
        attempts += 1
        if attempts > int(samples) * 20:
            raise RuntimeError(f"Could not collect {samples} inference states for lattice_size={shape}")

    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(rows), int(batch_size)):
        chunk = rows[start:start + int(batch_size)]
        keys = chunk[0].keys()
        batches.append({key: torch.cat([row[key] for row in chunk], dim=0) for key in keys})
    return batches


def collect_macro_runtime(
    *,
    model: mod.MacroDreamerEditModel,
    ckpt_args: dict[str, object],
    shape: tuple[int, int, int],
    segment_k: int,
    samples: int,
    candidate_limit: int,
    batch_size: int,
    repeats: int,
    warmup: int,
    device: str,
    amp_dtype: str,
    max_episode_steps: int,
    base_seed: int,
    temperature: float | None = None,
    cu_density: float | None = None,
) -> dict[str, float | int | list[float]]:
    batches = build_macro_batches(
        model=model,
        ckpt_args=ckpt_args,
        shape=shape,
        segment_k=segment_k,
        samples=samples,
        candidate_limit=candidate_limit,
        batch_size=batch_size,
        device=device,
        max_episode_steps=max_episode_steps,
        base_seed=base_seed,
        temperature=temperature,
        cu_density=cu_density,
    )
    model.eval()
    with torch.inference_mode():
        for _ in range(max(int(warmup), 0)):
            for tensors in batches:
                with autocast_context(device, amp_dtype):
                    forward_batch(model, tensors)
        synchronize(device)
        pass_times: list[float] = []
        for _ in range(max(int(repeats), 1)):
            t0 = time.perf_counter()
            for tensors in batches:
                with autocast_context(device, amp_dtype):
                    forward_batch(model, tensors)
            synchronize(device)
            pass_times.append(time.perf_counter() - t0)
    summary = summarize(pass_times)
    summary["segments"] = int(samples)
    summary["segment_k"] = int(segment_k)
    summary["segments_per_wall_second"] = float(int(samples) / max(float(summary["mean"]), 1e-12))
    summary["equivalent_micro_events_per_wall_second"] = float((int(samples) * int(segment_k)) / max(float(summary["mean"]), 1e-12))
    summary["batch_size"] = int(batch_size)
    summary["candidate_limit"] = int(candidate_limit)
    summary["timing_scope"] = "inference"
    summary["candidate_construction_included"] = False
    summary["projection_included"] = False
    return summary


def collect_macro_end_to_end_runtime(
    *,
    model: mod.MacroDreamerEditModel,
    ckpt_args: dict[str, object],
    shape: tuple[int, int, int],
    segment_k: int,
    samples: int,
    candidate_limit: int,
    repeats: int,
    warmup: int,
    device: str,
    amp_dtype: str,
    max_episode_steps: int,
    base_seed: int,
    temperature: float | None = None,
    cu_density: float | None = None,
) -> dict[str, float | int | list[float]]:
    cfg = build_env_cfg(
        ckpt_args,
        shape=shape,
        max_episode_steps=max_episode_steps,
        temperature=temperature,
        cu_density=cu_density,
    )
    max_seed_vacancies = int(ckpt_args["max_seed_vacancies"])
    run_times: list[float] = []
    completed_counts: list[int] = []
    model.eval()
    with torch.inference_mode():
        for repeat_idx in range(max(int(repeats), 1)):
            np.random.seed(base_seed + repeat_idx)
            torch.manual_seed(base_seed + repeat_idx)
            rng = np.random.default_rng(base_seed + repeat_idx)
            env = mod.MacroKMCEnv(cfg)
            env.reset()

            warm_completed = 0
            warm_attempts = 0
            while warm_completed < max(int(warmup), 0):
                tensors = long_eval._build_inference_tensors(
                    env=env,
                    max_seed_vacancies=max_seed_vacancies,
                    max_candidate_sites=int(candidate_limit),
                    horizon_k=int(segment_k),
                    device=device,
                )
                if tensors is not None:
                    with autocast_context(device, amp_dtype):
                        forward_batch(model, tensors, include_projection=True)
                    warm_completed += 1
                segment = long_eval._collect_teacher_segment(env, horizon_k=int(segment_k), rng=rng)
                if segment is None:
                    env = mod.MacroKMCEnv(cfg)
                    env.reset()
                warm_attempts += 1
                if warm_attempts > max(int(warmup), 1) * 50:
                    raise RuntimeError(f"Could not warm up end-to-end timing for lattice_size={shape}")
            synchronize(device)

            completed = 0
            attempts = 0
            elapsed = 0.0
            while completed < int(samples):
                t0 = time.perf_counter()
                tensors = long_eval._build_inference_tensors(
                    env=env,
                    max_seed_vacancies=max_seed_vacancies,
                    max_candidate_sites=int(candidate_limit),
                    horizon_k=int(segment_k),
                    device=device,
                )
                if tensors is not None:
                    with autocast_context(device, amp_dtype):
                        forward_batch(model, tensors, include_projection=True)
                    synchronize(device)
                    elapsed += time.perf_counter() - t0
                    completed += 1
                segment = long_eval._collect_teacher_segment(env, horizon_k=int(segment_k), rng=rng)
                if segment is None:
                    env = mod.MacroKMCEnv(cfg)
                    env.reset()
                attempts += 1
                if attempts > int(samples) * 50:
                    raise RuntimeError(f"Could not collect {samples} end-to-end inference states for lattice_size={shape}")
            run_times.append(elapsed)
            completed_counts.append(completed)

    summary = summarize(run_times)
    summary["segments"] = int(samples)
    summary["segment_k"] = int(segment_k)
    summary["segments_per_wall_second"] = float(int(samples) / max(float(summary["mean"]), 1e-12))
    summary["equivalent_micro_events_per_wall_second"] = float((int(samples) * int(segment_k)) / max(float(summary["mean"]), 1e-12))
    summary["batch_size"] = 1
    summary["candidate_limit"] = int(candidate_limit)
    summary["timing_scope"] = "end_to_end"
    summary["candidate_construction_included"] = True
    summary["tensor_creation_included"] = True
    summary["projection_included"] = True
    summary["state_application_included"] = False
    summary["sampling_teacher_advance_included"] = False
    summary["completed_counts"] = [int(x) for x in completed_counts]
    return summary


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    ckpt_args = ckpt["args"]
    model = long_eval._build_model(ckpt, args.device)

    rows = []
    lattice_shapes = (
        [parse_shape(token) for token in args.lattice_shapes]
        if args.lattice_shapes
        else [(int(edge), int(edge), int(edge)) for edge in args.lattice_edges]
    )
    for idx, shape in enumerate(lattice_shapes):
        lattice_l3 = shape_product(shape)
        shape_label = "x".join(str(int(dim)) for dim in shape)
        print(f"[lattice-size] shape={shape_label} L3={lattice_l3} start", flush=True)
        teacher, micro_events = collect_teacher_runtime(
            ckpt_args=ckpt_args,
            shape=shape,
            segment_k=int(args.segment_k),
            samples=int(args.samples),
            repeats=int(args.teacher_repeats),
            warmup_segments=int(args.teacher_warmup_segments),
            max_episode_steps=int(args.max_episode_steps_override),
            base_seed=int(args.seed) + idx * 1000,
            temperature=args.temperature,
            cu_density=args.cu_density,
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
                base_seed=int(args.seed) + idx * 1000 + 97,
                temperature=args.temperature,
                cu_density=args.cu_density,
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
                base_seed=int(args.seed) + idx * 1000 + 97,
                temperature=args.temperature,
                cu_density=args.cu_density,
            )
        speedup = float(float(teacher["mean"]) / max(float(macro["mean"]), 1e-12))
        row = {
            "lattice_shape": [int(dim) for dim in shape],
            "lattice_shape_label": shape_label,
            "lattice_edge": int(shape[0]) if shape[0] == shape[1] == shape[2] else None,
            "lattice_size_l3": int(lattice_l3),
            "bcc_sites": 2 * int(lattice_l3),
            "temperature": float(args.temperature) if args.temperature is not None else float(ckpt_args.get("temperature", 0.0)),
            "cu_density": float(args.cu_density) if args.cu_density is not None else float(ckpt_args.get("cu_density", 0.0)),
            "samples": int(args.samples),
            "segment_k": int(args.segment_k),
            "teacher_represented_micro_events": int(micro_events),
            "akmc_runtime_s": teacher,
            "atomworld_runtime_s": macro,
            "speedup_akmc_over_atomworld": speedup,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "lattice_shape": shape_label,
                    "L3": int(lattice_l3),
                    "akmc_runtime_s": float(teacher["mean"]),
                    "atomworld_runtime_s": float(macro["mean"]),
                    "speedup": speedup,
                },
                indent=2,
            ),
            flush=True,
        )

    output = {
        "mode": "lattice_size_runtime_scaling_fixed_horizon_macro_forward",
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "amp_dtype": args.amp_dtype,
        "matmul_precision": args.matmul_precision,
        "seed": int(args.seed),
        "lattice_edges": [int(x) for x in args.lattice_edges],
        "lattice_shapes": [[int(dim) for dim in shape] for shape in lattice_shapes],
        "samples": int(args.samples),
        "segment_k": int(args.segment_k),
        "teacher_repeats": int(args.teacher_repeats),
        "teacher_warmup_segments": int(args.teacher_warmup_segments),
        "macro_repeats": int(args.macro_repeats),
        "macro_warmup": int(args.macro_warmup),
        "macro_timing_scope": args.macro_timing_scope,
        "batch_size": int(args.batch_size),
        "candidate_limit": int(args.candidate_limit),
        "temperature": float(args.temperature) if args.temperature is not None else None,
        "cu_density": float(args.cu_density) if args.cu_density is not None else None,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[lattice-size] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
