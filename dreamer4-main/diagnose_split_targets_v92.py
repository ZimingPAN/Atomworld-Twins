from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


FE_TYPE = 0
CU_TYPE = 1
V_TYPE = 2

REQUIRED_MASK_FIELDS = (
    "changed_mask",
    "teacher_touched_mask",
    "teacher_action_source_mask",
    "teacher_action_destination_mask",
    "teacher_action_rollout_changed_mask",
)


def _as_array(sample: dict[str, Any], key: str, *, dtype: Any | None = None) -> np.ndarray:
    value = sample.get(key)
    if value is None:
        raise KeyError(key)
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _mask(sample: dict[str, Any], key: str, valid: np.ndarray) -> np.ndarray:
    arr = _as_array(sample, key, dtype=np.float32)
    if arr.shape != valid.shape:
        raise ValueError(f"{key} shape {arr.shape} does not match candidate_mask shape {valid.shape}")
    return (arr > 0.5) & valid


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_sum(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.sum(np.asarray(values, dtype=np.float64)))


def _prf(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    tp = float(np.logical_and(pred, target).sum())
    pred_count = float(pred.sum())
    target_count = float(target.sum())
    precision = tp / pred_count if pred_count > 0 else 0.0
    recall = tp / target_count if target_count > 0 else 0.0
    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "pred_count": pred_count,
        "target_count": target_count,
    }


def _coverage(pred: np.ndarray, target: np.ndarray) -> float:
    target_count = float(np.asarray(target, dtype=bool).sum())
    if target_count <= 0:
        return 0.0
    return float(np.logical_and(pred, target).sum() / target_count)


def _count(mask: np.ndarray) -> float:
    return float(np.asarray(mask, dtype=bool).sum())


def _edge_support_sites(sample: dict[str, Any], valid: np.ndarray) -> np.ndarray:
    edge_indices = sample.get("teacher_action_edge_pair_indices")
    edge_mask = sample.get("teacher_action_edge_pair_mask")
    if edge_indices is None or edge_mask is None:
        return np.zeros_like(valid, dtype=bool)
    edge_indices_arr = np.asarray(edge_indices, dtype=np.int64)
    edge_mask_arr = np.asarray(edge_mask, dtype=np.float32) > 0.5
    support = np.zeros_like(valid, dtype=bool)
    for pair, enabled in zip(edge_indices_arr, edge_mask_arr):
        if not enabled:
            continue
        if pair.shape[0] < 2:
            continue
        source_idx = int(pair[0])
        dest_idx = int(pair[1])
        if 0 <= source_idx < valid.shape[0] and valid[source_idx]:
            support[source_idx] = True
        if 0 <= dest_idx < valid.shape[0] and valid[dest_idx]:
            support[dest_idx] = True
    return support


def _sequence_support_sites(sample: dict[str, Any], valid: np.ndarray) -> np.ndarray:
    seq_indices = sample.get("teacher_action_sequence_indices")
    seq_mask = sample.get("teacher_action_sequence_mask")
    if seq_indices is None or seq_mask is None:
        return np.zeros_like(valid, dtype=bool)
    seq_indices_arr = np.asarray(seq_indices, dtype=np.int64)
    seq_mask_arr = np.asarray(seq_mask, dtype=np.float32) > 0.5
    support = np.zeros_like(valid, dtype=bool)
    for pair, enabled in zip(seq_indices_arr, seq_mask_arr):
        if not enabled:
            continue
        if pair.shape[0] < 2:
            continue
        source_idx = int(pair[0])
        dest_idx = int(pair[1])
        if 0 <= source_idx < valid.shape[0] and valid[source_idx]:
            support[source_idx] = True
        if 0 <= dest_idx < valid.shape[0] and valid[dest_idx]:
            support[dest_idx] = True
    return support


def _record_metrics(row: dict[str, float], prefix: str, pred: np.ndarray, target: np.ndarray) -> None:
    metrics = _prf(pred, target)
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def _sample_row(sample: dict[str, Any], *, split: str, index: int, reward_improving_sign: float) -> dict[str, float | int | str]:
    missing = [key for key in REQUIRED_MASK_FIELDS if key not in sample or sample.get(key) is None]
    if missing:
        raise KeyError(f"sample {split}[{index}] missing required split-target fields: {missing}")
    valid = _as_array(sample, "candidate_mask", dtype=np.float32) > 0.5
    current_types = _as_array(sample, "current_types", dtype=np.int64)
    target_types = _as_array(sample, "target_types", dtype=np.int64)
    if current_types.shape != valid.shape or target_types.shape != valid.shape:
        raise ValueError(f"sample {split}[{index}] type masks do not match candidate_mask")

    stored_changed = _mask(sample, "changed_mask", valid)
    final_type_diff = (current_types != target_types) & valid
    path_touched = _mask(sample, "teacher_touched_mask", valid)
    action_source = _mask(sample, "teacher_action_source_mask", valid)
    action_destination = _mask(sample, "teacher_action_destination_mask", valid)
    action_endpoint = (action_source | action_destination) & valid
    action_rollout_final_diff = _mask(sample, "teacher_action_rollout_changed_mask", valid)
    edge_pair_support = _edge_support_sites(sample, valid)
    sequence_endpoint = _sequence_support_sites(sample, valid)

    current_vacancy = (current_types == V_TYPE) & valid
    target_vacancy = (target_types == V_TYPE) & valid
    vacancy_departure = current_vacancy & ~target_vacancy
    vacancy_arrival = target_vacancy & ~current_vacancy
    vacancy_displacement = vacancy_departure | vacancy_arrival
    atom_type_change = final_type_diff & ~vacancy_displacement

    reward_sum = float(sample.get("reward_sum", 0.0))
    reward_improving = reward_sum * reward_improving_sign > 0.0
    reward_improving_support = final_type_diff if reward_improving else np.zeros_like(final_type_diff, dtype=bool)
    reward_nonzero_support = final_type_diff if abs(reward_sum) > 1e-12 else np.zeros_like(final_type_diff, dtype=bool)

    planner_projected = None
    if sample.get("planner_projected_changed_mask") is not None:
        planner_projected = _mask(sample, "planner_projected_changed_mask", valid)
    planner_candidate_teacher = None
    if sample.get("planner_candidate_teacher_changed_mask") is not None:
        planner_candidate_teacher = _mask(sample, "planner_candidate_teacher_changed_mask", valid)
    planner_candidate_fp = None
    if sample.get("planner_candidate_false_positive_mask") is not None:
        planner_candidate_fp = _mask(sample, "planner_candidate_false_positive_mask", valid)

    row: dict[str, float | int | str] = {
        "split": split,
        "index": int(index),
        "horizon_k": int(sample.get("horizon_k", -1)),
        "reward_sum": reward_sum,
        "reward_improving": float(reward_improving),
        "candidate_count": _count(valid),
        "changed_mask_mismatch_count": _count(stored_changed ^ final_type_diff),
        "final_type_diff_count": _count(final_type_diff),
        "action_endpoint_support_count": _count(action_endpoint),
        "path_touched_support_count": _count(path_touched),
        "action_rollout_final_diff_count": _count(action_rollout_final_diff),
        "edge_pair_support_count": _count(edge_pair_support),
        "sequence_endpoint_count": _count(sequence_endpoint),
        "vacancy_displacement_count": _count(vacancy_displacement),
        "vacancy_departure_count": _count(vacancy_departure),
        "vacancy_arrival_count": _count(vacancy_arrival),
        "atom_type_change_count": _count(atom_type_change),
        "reward_improving_support_count": _count(reward_improving_support),
        "reward_nonzero_support_count": _count(reward_nonzero_support),
        "endpoint_extra_over_final_count": _count(action_endpoint & ~final_type_diff),
        "touched_extra_over_final_count": _count(path_touched & ~final_type_diff),
        "final_missing_from_endpoint_count": _count(final_type_diff & ~action_endpoint),
        "final_missing_from_touched_count": _count(final_type_diff & ~path_touched),
        "final_missing_from_rollout_count": _count(final_type_diff & ~action_rollout_final_diff),
        "rollout_extra_over_final_count": _count(action_rollout_final_diff & ~final_type_diff),
        "vacancy_missing_from_rollout_count": _count(vacancy_displacement & ~action_rollout_final_diff),
        "vacancy_missing_from_endpoint_count": _count(vacancy_displacement & ~action_endpoint),
        "vacancy_missing_from_touched_count": _count(vacancy_displacement & ~path_touched),
    }

    _record_metrics(row, "endpoint_vs_final", action_endpoint, final_type_diff)
    _record_metrics(row, "touched_vs_final", path_touched, final_type_diff)
    _record_metrics(row, "rollout_vs_final", action_rollout_final_diff, final_type_diff)
    _record_metrics(row, "vacancy_vs_final", vacancy_displacement, final_type_diff)
    _record_metrics(row, "endpoint_vs_vacancy", action_endpoint, vacancy_displacement)
    _record_metrics(row, "touched_vs_vacancy", path_touched, vacancy_displacement)
    _record_metrics(row, "rollout_vs_vacancy", action_rollout_final_diff, vacancy_displacement)
    _record_metrics(row, "edgepair_vs_final", edge_pair_support, final_type_diff)
    _record_metrics(row, "sequence_vs_final", sequence_endpoint, final_type_diff)

    row["final_type_diff_covered_by_endpoint_recall"] = _coverage(action_endpoint, final_type_diff)
    row["final_type_diff_covered_by_touched_recall"] = _coverage(path_touched, final_type_diff)
    row["final_type_diff_covered_by_rollout_recall"] = _coverage(action_rollout_final_diff, final_type_diff)
    row["vacancy_displacement_covered_by_endpoint_recall"] = _coverage(action_endpoint, vacancy_displacement)
    row["vacancy_displacement_covered_by_touched_recall"] = _coverage(path_touched, vacancy_displacement)
    row["vacancy_displacement_covered_by_rollout_recall"] = _coverage(action_rollout_final_diff, vacancy_displacement)
    row["atom_type_change_covered_by_endpoint_recall"] = _coverage(action_endpoint, atom_type_change)
    row["atom_type_change_covered_by_touched_recall"] = _coverage(path_touched, atom_type_change)
    row["atom_type_change_covered_by_rollout_recall"] = _coverage(action_rollout_final_diff, atom_type_change)

    if planner_projected is not None:
        _record_metrics(row, "planner_projected_vs_final", planner_projected, final_type_diff)
        _record_metrics(row, "planner_projected_vs_endpoint", planner_projected, action_endpoint)
        row["planner_projected_count"] = _count(planner_projected)
    if planner_candidate_teacher is not None:
        _record_metrics(row, "planner_candidate_teacher_vs_final", planner_candidate_teacher, final_type_diff)
        row["planner_candidate_teacher_count"] = _count(planner_candidate_teacher)
    if planner_candidate_fp is not None:
        row["planner_candidate_false_positive_count"] = _count(planner_candidate_fp)
        row["planner_candidate_fp_inside_endpoint_count"] = _count(planner_candidate_fp & action_endpoint)
        row["planner_candidate_fp_inside_touched_count"] = _count(planner_candidate_fp & path_touched)
        row["planner_candidate_fp_inside_final_count"] = _count(planner_candidate_fp & final_type_diff)

    return row


def _aggregate(rows: list[dict[str, float | int | str]]) -> dict[str, float | int | dict[str, Any]]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    summary: dict[str, float | int | dict[str, Any]] = {"samples": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        summary[f"{key}_mean"] = _safe_mean(values)
        summary[f"{key}_sum"] = _safe_sum(values)
    by_horizon: dict[str, dict[str, float | int]] = {}
    horizons = sorted({int(row.get("horizon_k", -1)) for row in rows})
    for horizon in horizons:
        horizon_rows = [row for row in rows if int(row.get("horizon_k", -1)) == horizon]
        by_horizon[str(horizon)] = {
            "samples": len(horizon_rows),
            "final_type_diff_count_mean": _safe_mean([float(row["final_type_diff_count"]) for row in horizon_rows]),
            "endpoint_vs_final_f1_mean": _safe_mean([float(row["endpoint_vs_final_f1"]) for row in horizon_rows]),
            "rollout_vs_final_f1_mean": _safe_mean([float(row["rollout_vs_final_f1"]) for row in horizon_rows]),
            "vacancy_vs_final_f1_mean": _safe_mean([float(row["vacancy_vs_final_f1"]) for row in horizon_rows]),
            "endpoint_extra_over_final_count_mean": _safe_mean(
                [float(row["endpoint_extra_over_final_count"]) for row in horizon_rows]
            ),
            "final_missing_from_rollout_count_mean": _safe_mean(
                [float(row["final_missing_from_rollout_count"]) for row in horizon_rows]
            ),
        }
    summary["by_horizon"] = by_horizon
    return summary


def _load_cache(path: Path) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict):
        raise TypeError(f"Expected dict cache, got {type(cache)!r}")
    if "train" not in cache or "val" not in cache:
        raise KeyError("Cache must contain train and val splits")
    return cache


def _diagnose(cache: dict[str, Any], *, reward_improving_sign: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "signature": cache.get("signature", {}),
        "split_target_names": {
            "action_endpoint_support": "teacher_action_source_mask OR teacher_action_destination_mask; use as support-compression gate only.",
            "path_touched_support": "teacher_touched_mask; use as path support-compression target only.",
            "final_type_diff": "current_types != target_types inside candidate_mask; current changed_mask should match this.",
            "vacancy_displacement": "vacancy-status XOR between current_types and target_types; terminal state-diff target.",
            "action_rollout_final_diff": "teacher_action_rollout_changed_mask from replayed ordered actions in candidate state.",
            "reward_improving_support": "final_type_diff gated by reward_sum sign; diagnostic energy-planner target.",
        },
        "splits": {},
        "alerts": [],
    }
    for split in ("train", "val"):
        rows = [
            _sample_row(sample, split=split, index=index, reward_improving_sign=reward_improving_sign)
            for index, sample in enumerate(cache.get(split, []))
        ]
        result["splits"][split] = {"summary": _aggregate(rows), "rows": rows}
    train_summary = result["splits"]["train"]["summary"]
    val_summary = result["splits"]["val"]["summary"]
    for split, summary in (("train", train_summary), ("val", val_summary)):
        if float(summary.get("changed_mask_mismatch_count_mean", 0.0)) > 0.0:
            result["alerts"].append(f"{split}: changed_mask no longer equals final_type_diff")
        if float(summary.get("endpoint_vs_final_recall_mean", 0.0)) >= 0.99:
            result["alerts"].append(f"{split}: endpoint support covers final_type_diff; do not train it as a distinct final-diff decoder")
        if float(summary.get("rollout_vs_final_f1_mean", 0.0)) < 0.2:
            result["alerts"].append(f"{split}: action_rollout_final_diff is not a small residual basis for final_type_diff")
        if float(summary.get("vacancy_vs_final_f1_mean", 0.0)) >= 0.9:
            result["alerts"].append(f"{split}: final_type_diff is mostly vacancy displacement")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only v92 split-target diagnostics for macro-edit caches.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--reward_improving_sign", type=float, default=1.0,
                        help="Reward sign treated as improving for reward_improving_support. Default +1.")
    args = parser.parse_args()

    cache = _load_cache(args.cache)
    result = _diagnose(cache, reward_improving_sign=float(args.reward_improving_sign))
    result["cache"] = str(args.cache)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "stage_summary.json"
    detail_path = args.output_dir / "v92_split_target_diagnostics.json"
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    summary_path.write_text(text + "\n")
    detail_path.write_text(text + "\n")
    print(json.dumps({
        "output": str(summary_path),
        "alerts": result["alerts"],
        "train": result["splits"]["train"]["summary"],
        "val": result["splits"]["val"]["summary"],
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
