#!/usr/bin/env python3
"""Benchmark batched macro world-model inference on cached macro segments."""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import eval_macro_time_alignment as eval_mod
import train_dreamer_macro_edit as train_mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--candidate_limit", type=int, default=0)
    parser.add_argument("--rebuild_current_state_candidates", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp_dtype", choices=["none", "float16", "bfloat16"], default="none")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--teacher_reference_json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


NN1_OFFSETS = np.array(
    [
        [1, 1, 1],
        [-1, -1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
    ],
    dtype=np.int32,
)


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def forward_batch(model, tensors: dict[str, torch.Tensor]) -> None:
    global_latent = model.encode_global(tensors["start_obs"])
    prior_mu, prior_logvar = model.prior_stats(
        global_latent,
        tensors["global_summary"],
        tensors["horizon_k"],
    )
    path_latent = model.sample_path_latent(prior_mu, prior_logvar, deterministic=True)
    next_pred = model.predict_next_global(global_latent, path_latent, tensors["horizon_k"])
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
    change_logits, raw_type_logits = model.decode_edit(
        site_latent=site_latent,
        patch_latent=patch_latent,
        predicted_next_global=next_pred,
        path_latent=path_latent,
        horizon_k=tensors["horizon_k"],
        current_types=tensors["current_types"],
    )
    train_mod._predict_reward_and_duration_outputs(
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


def autocast_context(device: str, amp_dtype: str):
    if amp_dtype == "none" or not device.startswith("cuda") or not torch.cuda.is_available():
        return contextlib.nullcontext()
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def apply_candidate_limit(tensors: dict[str, torch.Tensor], limit: int) -> dict[str, torch.Tensor]:
    if limit <= 0:
        return tensors
    candidate_keys = {
        "candidate_positions",
        "nearest_vacancy_offset",
        "reach_depth",
        "is_start_vacancy",
        "current_types",
        "target_types",
        "change_targets",
        "candidate_mask",
    }
    limited = dict(tensors)
    for key in candidate_keys:
        value = limited.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            limited[key] = value[:, :limit, ...]
    return limited


def rebuild_current_state_candidates(samples: list, candidate_sites: int, max_seed_vacancies: int = 8) -> list:
    if candidate_sites <= 0:
        raise ValueError("--candidate_limit must be positive when rebuilding current-state candidates")
    rebuilt = []
    for sample in samples:
        box = np.asarray(sample.box_dims, dtype=np.int32)
        vacancies = np.asarray(sample.start_vacancy_positions, dtype=np.int32).reshape(-1, 3)
        seeds = vacancies[: max(1, min(max_seed_vacancies, len(vacancies)))]
        if seeds.size == 0:
            rebuilt.append(sample)
            continue

        depth_map: dict[tuple[int, int, int], int] = {}
        frontier = {tuple(map(int, pos.tolist())) for pos in seeds}
        for pos in frontier:
            depth_map[pos] = 0
        for depth in range(1, int(sample.horizon_k) + 1):
            next_frontier: set[tuple[int, int, int]] = set()
            for pos in frontier:
                for nxt in train_mod._one_hop_neighbors(pos, NN1_OFFSETS, box):
                    if nxt not in depth_map:
                        depth_map[nxt] = depth
                        next_frontier.add(nxt)
            frontier = next_frontier
            if not frontier:
                break

        def rank_key(pos: tuple[int, int, int]) -> tuple[int, float]:
            pos_arr = np.asarray(pos, dtype=np.float32)
            min_dist = min(
                np.linalg.norm(train_mod._periodic_offset(pos_arr, seed.astype(np.float32), box.astype(np.float32)))
                for seed in seeds
            )
            return depth_map[pos], float(min_dist)

        candidate_positions = sorted(depth_map.keys(), key=rank_key)[:candidate_sites]
        start_vac_set, start_cu_set = train_mod._positions_to_type_lookup(
            vacancies,
            np.asarray(sample.start_cu_positions, dtype=np.int32).reshape(-1, 3),
        )
        positions, nearest_offsets, reach_depth, is_start_vacancy, current_types, target_types, changed_mask = train_mod._build_patch_features(
            candidate_positions=candidate_positions,
            depth_map=depth_map,
            seeds=seeds,
            start_vac_set=start_vac_set,
            start_cu_set=start_cu_set,
            end_vac_set=start_vac_set,
            end_cu_set=start_cu_set,
            max_candidate_sites=candidate_sites,
            box=box,
            horizon_k=int(sample.horizon_k),
        )
        candidate_mask = np.zeros((candidate_sites,), dtype=np.float32)
        candidate_mask[: len(candidate_positions)] = 1.0
        rebuilt.append(
            replace(
                sample,
                candidate_positions=positions,
                nearest_vacancy_offset=nearest_offsets,
                reach_depth=reach_depth,
                is_start_vacancy=is_start_vacancy,
                current_types=current_types,
                target_types=target_types,
                candidate_mask=candidate_mask,
                changed_mask=changed_mask,
            )
        )
    return rebuilt


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = eval_mod._build_model(ckpt, args.device)
    segment_ks = eval_mod._segment_ks_from_ckpt_args(ckpt["args"])
    samples, dataset_stats, cache_signature = eval_mod._load_samples(
        args.cache,
        args.split,
        args.limit,
        expected_segment_ks=segment_ks,
        expected_summary_horizon_k=eval_mod._summary_horizon_k_from_ckpt_args(ckpt["args"]),
    )
    if args.rebuild_current_state_candidates:
        samples = rebuild_current_state_candidates(samples, args.candidate_limit)
    loader = train_mod._build_loader(samples, batch_size=args.batch_size, shuffle=False)
    batches = [
        apply_candidate_limit(train_mod._batch_to_device(batch, args.device), 0 if args.rebuild_current_state_candidates else args.candidate_limit)
        for batch in loader
    ]

    model.eval()
    with torch.inference_mode():
        for _ in range(max(args.warmup, 0)):
            for tensors in batches:
                with autocast_context(args.device, args.amp_dtype):
                    forward_batch(model, tensors)
        synchronize(args.device)
        pass_times = []
        for _ in range(max(args.repeats, 1)):
            t0 = time.perf_counter()
            for tensors in batches:
                with autocast_context(args.device, args.amp_dtype):
                    forward_batch(model, tensors)
            synchronize(args.device)
            pass_times.append(time.perf_counter() - t0)

    pass_times_np = np.asarray(pass_times, dtype=np.float64)
    segments_per_pass = len(samples)
    macro_segments_per_second = segments_per_pass / max(float(pass_times_np.mean()), 1e-12)
    horizon_events = int(sum(int(sample.horizon_k) for sample in samples))
    macro_micro_events_per_second = horizon_events / max(float(pass_times_np.mean()), 1e-12)

    output = {
        "mode": "batched_macro_forward_inference_on_cached_segments",
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "split": args.split,
        "num_samples": int(len(samples)),
        "segment_ks": [int(k) for k in segment_ks],
        "horizon_micro_events_per_pass": int(horizon_events),
        "batch_size": int(args.batch_size),
        "candidate_limit": int(args.candidate_limit),
        "candidate_source": "current_state_active_region" if args.rebuild_current_state_candidates else "cached_candidate_tensors",
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "device": args.device,
        "amp_dtype": args.amp_dtype,
        "matmul_precision": args.matmul_precision,
        "macro_wall_s_per_pass": {
            "mean": float(pass_times_np.mean()),
            "min": float(pass_times_np.min()),
            "max": float(pass_times_np.max()),
            "std": float(pass_times_np.std()),
            "values": [float(x) for x in pass_times_np.tolist()],
        },
        "macro_segments_per_wall_second": float(macro_segments_per_second),
        "macro_equivalent_micro_events_per_wall_second": float(macro_micro_events_per_second),
        "dataset_stats": dataset_stats,
        "cache_signature": cache_signature,
    }

    if args.teacher_reference_json is not None and args.teacher_reference_json.exists():
        ref = json.loads(args.teacher_reference_json.read_text())
        summary = ref.get("summary", {})
        teacher_seg_s = float(summary.get("teacher_kmc_segments_per_wall_second", {}).get("mean", 0.0))
        teacher_evt_s = float(summary.get("teacher_micro_events_per_wall_second", {}).get("mean", 0.0))
        if teacher_seg_s > 0:
            output["speedup_vs_teacher_segments_per_second"] = float(macro_segments_per_second / teacher_seg_s)
        if teacher_evt_s > 0:
            output["speedup_vs_teacher_micro_events_per_second"] = float(macro_micro_events_per_second / teacher_evt_s)
        output["teacher_reference_json"] = str(args.teacher_reference_json)
        output["teacher_reference"] = {
            "teacher_kmc_segments_per_wall_second": teacher_seg_s,
            "teacher_micro_events_per_wall_second": teacher_evt_s,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
