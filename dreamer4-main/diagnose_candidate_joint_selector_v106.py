#!/usr/bin/env python3
"""Grouped read-only v106 diagnostic for v105 candidate samples.

v105 intentionally expands each closed-loop state across multiple projection
sources and pair budgets.  A plain leave-one-segment-out split over the expanded
records can leak the same underlying state through a different source variant.
This script re-runs the two-branch selector calibration with the original
source_segment_index as the held-out group.

It reads the v105 JSONL sample export only.  It does not train a checkpoint,
modify planner behavior, or run long evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from diagnose_candidate_joint_selector_v103 import FEATURE_NAMES, _summarize_records


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_record(sample: dict[str, object]) -> dict[str, object]:
    targets = sample.get("targets") if isinstance(sample.get("targets"), dict) else {}
    features = sample.get("features") if isinstance(sample.get("features"), dict) else {}
    record: dict[str, object] = {
        "source_label": sample.get("source_label", ""),
        "source_index": sample.get("source_index", 0),
        "source_segment_index": sample.get("source_segment_index", sample.get("segment_index", 0)),
        "expanded_segment_index": sample.get("segment_index", 0),
        "candidate_index": sample.get("candidate_index", 0),
        "segment_k": sample.get("segment_k", 0),
        "selected_by_planner": bool(sample.get("selected_by_planner", False)),
        "v106_energy_site_target": _as_float(targets.get("energy_site")),
        "v106_pair_precision_target": _as_float(targets.get("pair_precision")),
        "teacher_reward_sum": _as_float(targets.get("teacher_reward_sum")),
        "site_f1": _as_float(targets.get("site_f1")),
        "vacancy_pair_precision": _as_float(targets.get("vacancy_pair_precision")),
        "vacancy_pair_recall": _as_float(targets.get("vacancy_pair_recall")),
        "vacancy_pair_f1": _as_float(targets.get("vacancy_pair_f1")),
        "vacancy_pair_count_efficiency": _as_float(targets.get("vacancy_pair_count_efficiency")),
        "vacancy_pair_selected_count": _as_float(targets.get("vacancy_pair_selected_count")),
        "vacancy_pair_teacher_count": _as_float(targets.get("vacancy_pair_teacher_count")),
        "vacancy_pair_typed_endpoint_accuracy": _as_float(targets.get("typed_endpoint_accuracy")),
        "features": np.asarray([_as_float(features.get(name)) for name in FEATURE_NAMES], dtype=np.float64),
    }
    return record


def _load_samples(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(_as_record(json.loads(line)))
    return records


def _group_by_base_segment(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[int(_as_float(record.get("source_segment_index")))].append(record)
    return [
        {"base_segment_index": index, "records": grouped[index]}
        for index in sorted(grouped)
    ]


def _pick_by(records: list[dict[str, object]], key: str) -> dict[str, object]:
    return max(records, key=lambda record: _as_float(record.get(key)))


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo + 1.0e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def _feature_matrix(records: list[dict[str, object]]) -> np.ndarray:
    if not records:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack([record["features"] for record in records]).astype(np.float64)


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    if x.size == 0:
        return np.zeros((len(FEATURE_NAMES),), dtype=np.float64)
    xtx = x.T @ x
    reg = float(ridge) * np.eye(xtx.shape[0], dtype=np.float64)
    return np.linalg.solve(xtx + reg, x.T @ y)


def _grouped_loo_two_branch(
    groups: list[dict[str, object]],
    pair_weights: list[float],
    ridge: float,
) -> dict[str, object]:
    predictions_by_weight: dict[float, list[dict[str, object]]] = {float(w): [] for w in pair_weights}
    picked_details: dict[float, list[dict[str, object]]] = {float(w): [] for w in pair_weights}
    for heldout in groups:
        heldout_records = heldout["records"]
        train_records = [
            record
            for group in groups
            if group is not heldout
            for record in group["records"]
        ]
        x_train = _feature_matrix(train_records)
        y_energy = np.asarray(
            [_as_float(record.get("v106_energy_site_target")) for record in train_records],
            dtype=np.float64,
        )
        y_pair = np.asarray(
            [_as_float(record.get("v106_pair_precision_target")) for record in train_records],
            dtype=np.float64,
        )
        w_energy = _ridge_fit(x_train, y_energy, ridge)
        w_pair = _ridge_fit(x_train, y_pair, ridge)
        x_test = _feature_matrix(heldout_records)
        pred_energy = _normalize(x_test @ w_energy)
        pred_pair = _normalize(x_test @ w_pair)
        for pair_weight in pair_weights:
            score = pred_energy + float(pair_weight) * pred_pair
            best_idx = int(np.argmax(score))
            picked = heldout_records[best_idx]
            predictions_by_weight[float(pair_weight)].append(picked)
            picked_details[float(pair_weight)].append(
                {
                    "base_segment_index": int(heldout["base_segment_index"]),
                    "source_label": str(picked.get("source_label", "")),
                    "candidate_index": int(_as_float(picked.get("candidate_index"))),
                    "segment_k": int(_as_float(picked.get("segment_k"))),
                    "score": float(score[best_idx]),
                    "energy_pred_norm": float(pred_energy[best_idx]),
                    "pair_pred_norm": float(pred_pair[best_idx]),
                    "energy_site_target": _as_float(picked.get("v106_energy_site_target")),
                    "pair_precision_target": _as_float(picked.get("v106_pair_precision_target")),
                    "site_f1": _as_float(picked.get("site_f1")),
                    "vacancy_pair_precision": _as_float(picked.get("vacancy_pair_precision")),
                    "vacancy_pair_f1": _as_float(picked.get("vacancy_pair_f1")),
                    "teacher_reward_sum": _as_float(picked.get("teacher_reward_sum")),
                    "vacancy_pair_selected_count": _as_float(picked.get("vacancy_pair_selected_count")),
                }
            )
    return {
        "ridge": float(ridge),
        "feature_names": list(FEATURE_NAMES),
        "pair_weight_sweep": {
            str(float(pair_weight)): {
                "summary": _summarize_records(records),
                "picked_source_histogram": dict(
                    sorted(Counter(str(record.get("source_label", "")) for record in records).items())
                ),
                "picked": picked_details[float(pair_weight)],
            }
            for pair_weight, records in predictions_by_weight.items()
        },
    }


def _oracle_two_branch(
    groups: list[dict[str, object]],
    pair_weights: list[float],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for pair_weight in pair_weights:
        picked: list[dict[str, object]] = []
        for group in groups:
            records = group["records"]
            energy = _normalize(
                np.asarray([_as_float(record.get("v106_energy_site_target")) for record in records])
            )
            pair = _normalize(
                np.asarray([_as_float(record.get("v106_pair_precision_target")) for record in records])
            )
            score = energy + float(pair_weight) * pair
            picked.append(records[int(np.argmax(score))])
        output[str(float(pair_weight))] = _summarize_records(picked)
    return output


def _source_histogram(records: Iterable[dict[str, object]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get("source_label", "")) for record in records).items()))


def _slim(summary: dict[str, object]) -> dict[str, object]:
    keys = [
        "count",
        "avg_site_f1",
        "avg_vacancy_pair_precision",
        "avg_vacancy_pair_recall",
        "avg_vacancy_pair_f1",
        "avg_vacancy_pair_count_efficiency",
        "avg_teacher_reward_sum",
        "avg_projected_changed_count",
        "avg_vacancy_pair_selected_count",
    ]
    return {key: summary.get(key) for key in keys}


def run(
    samples_path: Path,
    output_path: Path,
    pair_weights: list[float],
    ridge: float,
) -> dict[str, object]:
    records = _load_samples(samples_path)
    groups = _group_by_base_segment(records)
    planner_selected = [record for record in records if bool(record.get("selected_by_planner", False))]
    base_uncapped_selected = [
        record
        for record in planner_selected
        if str(record.get("source_label", "")) == "uncapped_a32_d4"
    ]
    oracle = _oracle_two_branch(groups, pair_weights=pair_weights)
    grouped_loo = _grouped_loo_two_branch(groups, pair_weights=pair_weights, ridge=ridge)
    pair_counts = [_as_float(record.get("vacancy_pair_selected_count")) for record in records]
    result = {
        "mode": "v106_grouped_base_segment_two_branch_readonly",
        "samples_path": str(samples_path),
        "sample_count": int(len(records)),
        "base_segment_count": int(len(groups)),
        "expanded_selected_count": int(len(planner_selected)),
        "pair_selected_count_unique": sorted({float(value) for value in pair_counts}),
        "pair_selected_count_std": float(np.std(pair_counts)) if pair_counts else 0.0,
        "source_histogram": _source_histogram(records),
        "planner_selected_expanded": _summarize_records(planner_selected),
        "planner_selected_uncapped_only": _summarize_records(base_uncapped_selected),
        "oracle_energy_site_base": _summarize_records(
            [_pick_by(group["records"], "v106_energy_site_target") for group in groups]
        ),
        "oracle_pair_precision_base": _summarize_records(
            [_pick_by(group["records"], "v106_pair_precision_target") for group in groups]
        ),
        "oracle_two_branch_base": oracle,
        "grouped_leave_one_base_segment_out_two_branch": grouped_loo,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair_weights", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    args = parser.parse_args()
    result = run(
        samples_path=args.samples,
        output_path=args.output,
        pair_weights=[float(v) for v in args.pair_weights],
        ridge=float(args.ridge),
    )
    loo = result["grouped_leave_one_base_segment_out_two_branch"]["pair_weight_sweep"]
    summary = {
        "mode": result["mode"],
        "sample_count": result["sample_count"],
        "base_segment_count": result["base_segment_count"],
        "pair_selected_count_unique": result["pair_selected_count_unique"],
        "pair_selected_count_std": result["pair_selected_count_std"],
        "planner_selected_expanded": _slim(result.get("planner_selected_expanded", {})),
        "planner_selected_uncapped_only": _slim(result.get("planner_selected_uncapped_only", {})),
        "oracle_two_branch_pair_weight_1": _slim(result.get("oracle_two_branch_base", {}).get("1.0", {})),
        "grouped_loo_pair_weight_0": _slim(loo.get("0.0", {}).get("summary", {})),
        "grouped_loo_pair_weight_1": _slim(loo.get("1.0", {}).get("summary", {})),
        "grouped_loo_pair_weight_2": _slim(loo.get("2.0", {}).get("summary", {})),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
