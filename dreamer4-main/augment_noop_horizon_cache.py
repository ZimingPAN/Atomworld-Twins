#!/usr/bin/env python3
"""Augment cached no-op macro segments across requested horizons.

This is a diagnostic data transform for planner-selected hard negatives.  It
keeps normal segments unchanged and duplicates zero-edit / zero-reward no-op
samples onto missing horizon IDs so reward/edit heads see no-op states at the
same Multi-K support used by long-rollout planning.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def _parse_horizons(text: str) -> list[int]:
    values = [int(item) for item in text.replace(",", " ").split() if item.strip()]
    if not values:
        raise ValueError("at least one horizon is required")
    return sorted(set(values))


def _is_noop(item: dict) -> bool:
    changed = np.asarray(item["changed_mask"])
    return float(changed.sum()) <= 0.0 and abs(float(item.get("reward_sum", 0.0))) <= 1e-12


def _retarget_noop(item: dict, new_horizon: int, summary_horizon: int) -> dict:
    out = copy.deepcopy(item)
    old_horizon = max(int(out.get("horizon_k", new_horizon)), 1)
    new_horizon = int(new_horizon)
    out["horizon_k"] = new_horizon

    scale = float(new_horizon) / float(old_horizon)
    out["tau_exp"] = float(out.get("tau_exp", 0.0)) * scale
    out["tau_real"] = float(out.get("tau_real", out.get("tau_exp", 0.0))) * scale
    out["reward_sum"] = 0.0
    out["target_types"] = np.asarray(out["current_types"]).copy()
    out["changed_mask"] = np.zeros_like(np.asarray(out["changed_mask"], dtype=np.float32))

    summary = np.asarray(out["teacher_path_summary"], dtype=np.float32).copy()
    if summary.shape[0] >= 18 + 2 * int(summary_horizon):
        step_log = summary[18 : 18 + int(summary_horizon)]
        observed = step_log[:old_horizon]
        fill = float(np.median(observed)) if observed.size else -27.0
        step_log[:] = -27.0
        step_log[: min(new_horizon, int(summary_horizon))] = fill
        step_delta = summary[18 + int(summary_horizon) : 18 + 2 * int(summary_horizon)]
        step_delta[:] = 0.0
        summary[17] = 1.0
    out["teacher_path_summary"] = summary
    return out


def _augment_split(samples: list[dict], horizons: list[int], summary_horizon: int) -> tuple[list[dict], dict]:
    augmented = list(samples)
    noop_by_horizon = Counter()
    added_by_horizon = Counter()
    for item in samples:
        if not _is_noop(item):
            continue
        old_horizon = int(item["horizon_k"])
        noop_by_horizon[old_horizon] += 1
        for horizon in horizons:
            if horizon == old_horizon:
                continue
            augmented.append(_retarget_noop(item, horizon, summary_horizon))
            added_by_horizon[horizon] += 1
    final_noop_by_horizon = Counter()
    for item in augmented:
        if _is_noop(item):
            final_noop_by_horizon[int(item["horizon_k"])] += 1
    return augmented, {
        "original_samples": len(samples),
        "augmented_samples": len(augmented),
        "original_noop_by_horizon": dict(sorted(noop_by_horizon.items())),
        "added_noop_by_horizon": dict(sorted(added_by_horizon.items())),
        "final_noop_by_horizon": dict(sorted(final_noop_by_horizon.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizons", default="128,256,512,1024")
    parser.add_argument("--summary_horizon", type=int, default=None)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    horizons = _parse_horizons(args.horizons)
    signature = dict(payload.get("signature") or {})
    # Caches produced before the hard-negative follow-up predate this signature
    # key.  The augmented cache is meant to be consumed by the current training
    # script with the default value, so preserve compatibility explicitly.
    signature.setdefault("keep_after_noop_segments", False)
    summary_horizon = int(args.summary_horizon or signature.get("summary_horizon_k") or max(horizons))

    output = dict(payload)
    stats = dict(payload.get("stats") or {})
    augmentation_stats = {}
    for split in ("train", "val"):
        split_samples = [dict(item) for item in payload[split]]
        output[split], augmentation_stats[split] = _augment_split(split_samples, horizons, summary_horizon)
    output["stats"] = stats
    output["noop_horizon_augmentation"] = {
        "horizons": horizons,
        "summary_horizon": summary_horizon,
        **augmentation_stats,
    }
    output["signature"] = signature

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps(output["noop_horizon_augmentation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
