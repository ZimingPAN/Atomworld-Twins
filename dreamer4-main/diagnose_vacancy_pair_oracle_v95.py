from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


FE_TYPE = 0
CU_TYPE = 1
V_TYPE = 2


def _as_array(sample: dict[str, Any], key: str, *, dtype: Any | None = None) -> np.ndarray:
    value = sample.get(key)
    if value is None:
        raise KeyError(key)
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _safe_sum(values: list[float]) -> float:
    return float(np.sum(np.asarray(values, dtype=np.float64))) if values else 0.0


def _count(mask: np.ndarray) -> float:
    return float(np.asarray(mask, dtype=bool).sum())


def _prf(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_bool = np.asarray(pred, dtype=bool)
    target_bool = np.asarray(target, dtype=bool)
    tp = float(np.logical_and(pred_bool, target_bool).sum())
    pred_count = float(pred_bool.sum())
    target_count = float(target_bool.sum())
    precision = tp / pred_count if pred_count > 0 else 0.0
    recall = tp / target_count if target_count > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "pred_count": float(pred_count),
        "target_count": float(target_count),
    }


def _type_accuracy(pred_types: np.ndarray, target_types: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask, dtype=bool)
    denom = int(mask_bool.sum())
    if denom <= 0:
        return 0.0
    return float((np.asarray(pred_types)[mask_bool] == np.asarray(target_types)[mask_bool]).mean())


def _record_prf(row: dict[str, float], prefix: str, pred: np.ndarray, target: np.ndarray) -> None:
    for key, value in _prf(pred, target).items():
        row[f"{prefix}_{key}"] = value


def _sequence_pairs(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = sample.get("teacher_action_sequence_indices")
    mask = sample.get("teacher_action_sequence_mask")
    moving_type = sample.get("teacher_action_sequence_moving_type")
    order = sample.get("teacher_action_sequence_order")
    if indices is None or mask is None:
        indices = sample.get("teacher_action_edge_pair_indices")
        mask = sample.get("teacher_action_edge_pair_mask")
        moving_type = sample.get("teacher_action_edge_pair_moving_type")
        order = sample.get("teacher_action_edge_pair_order")
    if indices is None or mask is None:
        raise KeyError("sample missing teacher action sequence / edge pair fields")
    indices_arr = np.asarray(indices, dtype=np.int64)
    mask_arr = np.asarray(mask, dtype=np.float32) > 0.5
    if moving_type is None:
        moving_arr = np.full((indices_arr.shape[0],), -1, dtype=np.int64)
    else:
        moving_arr = np.asarray(moving_type, dtype=np.int64)
    if order is None:
        order_arr = np.arange(indices_arr.shape[0], dtype=np.float32)
    else:
        order_arr = np.asarray(order, dtype=np.float32)
    enabled = np.flatnonzero(mask_arr)
    if enabled.size <= 0:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            enabled,
        )
    order_idx = enabled[np.argsort(order_arr[enabled], kind="stable")]
    return indices_arr[order_idx], moving_arr[order_idx], order_arr[order_idx], order_idx


def _rollout_types(
    *,
    current_types: np.ndarray,
    pairs: np.ndarray,
    moving_types: np.ndarray,
    valid: np.ndarray,
    orientation: str,
) -> np.ndarray:
    state = np.asarray(current_types, dtype=np.int64).copy()
    valid_bool = np.asarray(valid, dtype=bool)
    for pair, moving in zip(pairs, moving_types):
        if pair.shape[0] < 2:
            continue
        old_idx = int(pair[0])
        new_idx = int(pair[1])
        if old_idx < 0 or new_idx < 0 or old_idx >= state.shape[0] or new_idx >= state.shape[0]:
            continue
        if not valid_bool[old_idx] or not valid_bool[new_idx]:
            continue
        if orientation == "kmc_vacancy_to_atom":
            moved = int(moving)
            if moved not in (FE_TYPE, CU_TYPE):
                moved = int(state[new_idx])
                if moved == V_TYPE:
                    moved = FE_TYPE
            state[old_idx] = moved
            state[new_idx] = V_TYPE
        elif orientation == "legacy_atom_to_vacancy":
            moved = int(moving)
            if moved not in (FE_TYPE, CU_TYPE):
                moved = int(state[old_idx])
                if moved == V_TYPE:
                    moved = FE_TYPE
            state[old_idx] = V_TYPE
            state[new_idx] = moved
        else:
            raise ValueError(f"unknown orientation: {orientation}")
    return state


def _pair_orientation_stats(
    *,
    current_types: np.ndarray,
    target_types: np.ndarray,
    pairs: np.ndarray,
    moving_types: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    if pairs.shape[0] == 0:
        return {
            "pair_count": 0.0,
            "old_current_vacancy_frac": 0.0,
            "new_current_vacancy_frac": 0.0,
            "old_target_vacancy_frac": 0.0,
            "new_target_vacancy_frac": 0.0,
            "old_target_matches_moving_type_frac": 0.0,
            "new_target_matches_moving_type_frac": 0.0,
            "kmc_pair_typed_accuracy": 0.0,
            "legacy_pair_typed_accuracy": 0.0,
        }
    old_idx = pairs[:, 0].astype(np.int64)
    new_idx = pairs[:, 1].astype(np.int64)
    ok = (
        (old_idx >= 0)
        & (new_idx >= 0)
        & (old_idx < current_types.shape[0])
        & (new_idx < current_types.shape[0])
        & valid[old_idx]
        & valid[new_idx]
    )
    if not np.any(ok):
        return {
            "pair_count": float(pairs.shape[0]),
            "old_current_vacancy_frac": 0.0,
            "new_current_vacancy_frac": 0.0,
            "old_target_vacancy_frac": 0.0,
            "new_target_vacancy_frac": 0.0,
            "old_target_matches_moving_type_frac": 0.0,
            "new_target_matches_moving_type_frac": 0.0,
            "kmc_pair_typed_accuracy": 0.0,
            "legacy_pair_typed_accuracy": 0.0,
        }
    old_idx = old_idx[ok]
    new_idx = new_idx[ok]
    moving = moving_types[ok]
    valid_moving = np.isin(moving, [FE_TYPE, CU_TYPE])
    kmc_old_ok = target_types[old_idx] == moving
    kmc_new_ok = target_types[new_idx] == V_TYPE
    legacy_old_ok = target_types[old_idx] == V_TYPE
    legacy_new_ok = target_types[new_idx] == moving
    return {
        "pair_count": float(old_idx.shape[0]),
        "old_current_vacancy_frac": float((current_types[old_idx] == V_TYPE).mean()),
        "new_current_vacancy_frac": float((current_types[new_idx] == V_TYPE).mean()),
        "old_target_vacancy_frac": float((target_types[old_idx] == V_TYPE).mean()),
        "new_target_vacancy_frac": float((target_types[new_idx] == V_TYPE).mean()),
        "old_target_matches_moving_type_frac": float((kmc_old_ok & valid_moving).mean()),
        "new_target_matches_moving_type_frac": float((legacy_new_ok & valid_moving).mean()),
        "kmc_pair_typed_accuracy": float(((kmc_old_ok & valid_moving).astype(np.float32) + kmc_new_ok.astype(np.float32)).mean() / 2.0),
        "legacy_pair_typed_accuracy": float((legacy_old_ok.astype(np.float32) + (legacy_new_ok & valid_moving).astype(np.float32)).mean() / 2.0),
    }


def _sample_row(sample: dict[str, Any], split: str, index: int) -> dict[str, float | int | str]:
    valid = _as_array(sample, "candidate_mask", dtype=np.float32) > 0.5
    current_types = _as_array(sample, "current_types", dtype=np.int64)
    target_types = _as_array(sample, "target_types", dtype=np.int64)
    changed = (current_types != target_types) & valid
    current_vacancy = (current_types == V_TYPE) & valid
    target_vacancy = (target_types == V_TYPE) & valid
    vacancy_displacement = current_vacancy ^ target_vacancy
    vacancy_departure = current_vacancy & ~target_vacancy
    vacancy_arrival = target_vacancy & ~current_vacancy
    pairs, moving_types, _orders, _enabled = _sequence_pairs(sample)

    kmc_state = _rollout_types(
        current_types=current_types,
        pairs=pairs,
        moving_types=moving_types,
        valid=valid,
        orientation="kmc_vacancy_to_atom",
    )
    legacy_state = _rollout_types(
        current_types=current_types,
        pairs=pairs,
        moving_types=moving_types,
        valid=valid,
        orientation="legacy_atom_to_vacancy",
    )
    kmc_changed = (kmc_state != current_types) & valid
    legacy_changed = (legacy_state != current_types) & valid
    endpoint_mask = np.zeros_like(valid, dtype=bool)
    if pairs.size > 0:
        old_idx = pairs[:, 0]
        new_idx = pairs[:, 1]
        ok_old = (old_idx >= 0) & (old_idx < valid.shape[0])
        ok_new = (new_idx >= 0) & (new_idx < valid.shape[0])
        endpoint_mask[old_idx[ok_old]] |= valid[old_idx[ok_old]]
        endpoint_mask[new_idx[ok_new]] |= valid[new_idx[ok_new]]

    row: dict[str, float | int | str] = {
        "split": split,
        "index": int(index),
        "horizon_k": int(sample.get("horizon_k", -1)),
        "reward_sum": float(sample.get("reward_sum", 0.0)),
        "tau_exp": float(sample.get("tau_exp", 0.0)),
        "candidate_count": _count(valid),
        "final_type_diff_count": _count(changed),
        "vacancy_displacement_count": _count(vacancy_displacement),
        "vacancy_departure_count": _count(vacancy_departure),
        "vacancy_arrival_count": _count(vacancy_arrival),
        "sequence_pair_count": float(pairs.shape[0]),
        "endpoint_support_count": _count(endpoint_mask),
        "kmc_rollout_changed_count": _count(kmc_changed),
        "legacy_rollout_changed_count": _count(legacy_changed),
        "kmc_rollout_type_acc_on_final": _type_accuracy(kmc_state, target_types, changed),
        "legacy_rollout_type_acc_on_final": _type_accuracy(legacy_state, target_types, changed),
        "kmc_rollout_type_acc_on_vacancy": _type_accuracy(kmc_state, target_types, vacancy_displacement),
        "legacy_rollout_type_acc_on_vacancy": _type_accuracy(legacy_state, target_types, vacancy_displacement),
        "kmc_exact_all_valid_frac": float((kmc_state[valid] == target_types[valid]).mean()) if np.any(valid) else 0.0,
        "legacy_exact_all_valid_frac": float((legacy_state[valid] == target_types[valid]).mean()) if np.any(valid) else 0.0,
    }
    for key, value in _pair_orientation_stats(
        current_types=current_types,
        target_types=target_types,
        pairs=pairs,
        moving_types=moving_types,
        valid=valid,
    ).items():
        row[key] = value
    _record_prf(row, "endpoint_vs_final", endpoint_mask, changed)
    _record_prf(row, "endpoint_vs_vacancy", endpoint_mask, vacancy_displacement)
    _record_prf(row, "vacancy_oracle_vs_final", vacancy_displacement, changed)
    _record_prf(row, "kmc_rollout_vs_final", kmc_changed, changed)
    _record_prf(row, "kmc_rollout_vs_vacancy", kmc_changed, vacancy_displacement)
    _record_prf(row, "legacy_rollout_vs_final", legacy_changed, changed)
    _record_prf(row, "legacy_rollout_vs_vacancy", legacy_changed, vacancy_displacement)
    return row


def _summarize_rows(rows: list[dict[str, float | int | str]]) -> dict[str, float | int | dict[str, int]]:
    numeric: dict[str, list[float]] = {}
    hist: dict[str, int] = {}
    for row in rows:
        hist[str(int(row.get("horizon_k", -1)))] = hist.get(str(int(row.get("horizon_k", -1))), 0) + 1
        for key, value in row.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric.setdefault(key, []).append(float(value))
    summary: dict[str, float | int | dict[str, int]] = {
        "samples": len(rows),
        "horizon_histogram": hist,
    }
    for key, values in sorted(numeric.items()):
        if key in {"index", "horizon_k"}:
            continue
        summary[f"{key}_mean"] = _safe_mean(values)
        if key.endswith("_count") or key.endswith("_sum"):
            summary[f"{key}_sum"] = _safe_sum(values)
    return summary


def _load_stage_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = json.loads(path.read_text())
    files = {}
    for name, item in data.get("files", {}).items():
        cumulative = item.get("cumulative", {}) if isinstance(item, dict) else {}
        tau_expected = item.get("tau_expected", {}) if isinstance(item, dict) else {}
        files[name] = {
            "completed_rollout_segments": item.get("completed_rollout_segments"),
            "stop_reason": item.get("stop_reason"),
            "chosen_k_histogram": item.get("chosen_k_histogram"),
            "delta_e_ratio": cumulative.get("delta_e_ratio"),
            "expected_time_ratio": cumulative.get("expected_time_ratio"),
            "tau_scale_ratio": tau_expected.get("scale_ratio"),
            "selected_site_f1_mean": item.get("selected_site_f1_mean"),
            "projected_changed_count_mean": item.get("projected_changed_count_mean"),
            "teacher_changed_count_mean": item.get("teacher_changed_count_mean"),
            "planner_projection_change_source": item.get("planner_projection_change_source"),
            "planner_projection_topk_source": item.get("planner_projection_topk_source"),
            "planner_projection_topk_budget": item.get("planner_projection_topk_budget"),
        }
    return {"path": str(path), "exists": True, "files": files}


def _interpret(splits: dict[str, Any], stage_metrics: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    val = splits.get("val", {})
    kmc_f1 = float(val.get("kmc_rollout_vs_final_f1_mean", 0.0))
    legacy_f1 = float(val.get("legacy_rollout_vs_final_f1_mean", 0.0))
    kmc_type = float(val.get("kmc_rollout_type_acc_on_vacancy_mean", 0.0))
    legacy_type = float(val.get("legacy_rollout_type_acc_on_vacancy_mean", 0.0))
    if kmc_f1 > legacy_f1 + 0.25 and kmc_type > legacy_type + 0.25:
        notes.append(
            "teacher action pairs follow KMC vacancy->atom semantics; legacy atom->vacancy rollout orientation is inconsistent with cache targets."
        )
    elif legacy_f1 > kmc_f1 + 0.25:
        notes.append("legacy atom->vacancy orientation unexpectedly matches final targets better; inspect _decode_action semantics before patching.")
    else:
        notes.append("pair orientation is ambiguous from aggregate F1; inspect per-sample rows before patching rollout code.")

    vacancy_f1 = float(val.get("vacancy_oracle_vs_final_f1_mean", 0.0))
    endpoint_f1 = float(val.get("endpoint_vs_final_f1_mean", 0.0))
    if vacancy_f1 > 0.9 and endpoint_f1 > 0.9:
        notes.append("pair-level vacancy displacement and endpoint/touched support remain strong upper bounds for terminal support compression.")
    best_existing = 0.0
    for stage in stage_metrics.values():
        for item in stage.get("files", {}).values():
            f1 = item.get("selected_site_f1_mean")
            delta = item.get("delta_e_ratio")
            if f1 is not None and delta is not None and float(delta) >= 0.9:
                best_existing = max(best_existing, float(f1))
    if best_existing > 0.0 and best_existing < 0.25 and vacancy_f1 > 0.9:
        notes.append(
            "existing closed-loop pair/typed strategies preserve energy but remain far below the true vacancy-pair support upper bound."
        )
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only v95 pair-level vacancy-displacement oracle diagnostic.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage-summary", action="append", default=[])
    parser.add_argument("--max-rows", type=int, default=64)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"expected dict cache, got {type(data)!r}")

    splits: dict[str, Any] = {}
    rows_by_split: dict[str, list[dict[str, float | int | str]]] = {}
    for split in ("train", "val"):
        samples = data.get(split)
        if samples is None:
            continue
        rows = [_sample_row(sample, split, idx) for idx, sample in enumerate(samples)]
        rows_by_split[split] = rows
        splits[split] = _summarize_rows(rows)

    stage_metrics = {
        Path(path).parent.name or Path(path).stem: _load_stage_metrics(Path(path))
        for path in args.stage_summary
    }
    result = {
        "diagnostic": "v95_pair_level_vacancy_displacement_oracle",
        "cache": str(cache_path),
        "signature": data.get("signature", {}),
        "splits": splits,
        "stage_metrics": stage_metrics,
        "interpretation": _interpret(splits, stage_metrics),
        "rows_preview": {
            split: rows[: max(int(args.max_rows), 0)]
            for split, rows in rows_by_split.items()
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
