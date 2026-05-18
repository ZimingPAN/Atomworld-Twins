#!/usr/bin/env python3
"""Read-only v113 conditional compatibility calibration.

This diagnostic consumes v111 factorized vacancy-pair payloads and fits a small
leave-one-segment-out ridge scorer on model-visible pair factors. It does not
train or modify checkpoints. The goal is to test whether a simple calibrated
source-destination interaction score can move true terminal vacancy pairs into
usable top-k lists before adding another model head.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


BASE_SCORE_FIELDS = [
    "score",
    "vacancy_score",
    "energy_score",
    "source_score",
    "destination_score",
    "endpoint_sum_score",
    "interaction_residual",
    "moving_type_score",
    "order_early_score",
]
RANK_KEYS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
TOP_LABEL_KEYS = [
    "true_pair",
    "same_source_wrong_destination",
    "same_destination_wrong_source",
    "source_destination_unpaired",
    "false_pair",
]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _position_key(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return tuple(int(v) for v in value)
    except (TypeError, ValueError):
        return None


def _feature_vector(item: dict[str, Any]) -> list[float]:
    base = [_as_float(item.get(field, 0.0)) for field in BASE_SCORE_FIELDS]
    source = item.get("_source_key")
    dest = item.get("_destination_key")
    if isinstance(source, tuple) and isinstance(dest, tuple):
        diff = [float(dest[i] - source[i]) for i in range(3)]
    else:
        diff = [0.0, 0.0, 0.0]
    abs_diff = [abs(v) for v in diff]
    l1 = sum(abs_diff)
    l2 = math.sqrt(sum(v * v for v in diff))
    source_score = base[3]
    dest_score = base[4]
    vacancy_score = base[1]
    energy_score = base[2]
    moving_score = base[7]
    order_score = base[8]
    return [
        *base,
        source_score * dest_score,
        source_score - dest_score,
        dest_score - source_score,
        vacancy_score - energy_score,
        energy_score - vacancy_score,
        moving_score - order_score,
        *diff,
        *abs_diff,
        l1,
        l2,
    ]


def _candidate_records(data: dict[str, Any], source_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seg_index, segment in enumerate(data.get("segments", [])):
        if not isinstance(segment, dict):
            continue
        for cand_index, candidate in enumerate(segment.get("planner_candidates", [])):
            if not isinstance(candidate, dict):
                continue
            projection = candidate.get("vacancy_pair_projection_diagnostic")
            if not isinstance(projection, dict):
                continue
            pairs_raw = projection.get("factorized_pair_scores")
            if not isinstance(pairs_raw, list) or not pairs_raw:
                continue
            pairs: list[dict[str, Any]] = []
            for item in pairs_raw:
                if not isinstance(item, dict):
                    continue
                source = _position_key(item.get("source_position"))
                dest = _position_key(item.get("destination_position"))
                if source is None or dest is None:
                    continue
                pair = dict(item)
                pair["_source_key"] = source
                pair["_destination_key"] = dest
                pair["_features"] = _feature_vector(pair)
                pairs.append(pair)
            if not pairs:
                continue
            diagnostic = candidate.get("candidate_joint_diagnostic")
            if not isinstance(diagnostic, dict):
                diagnostic = {}
            oracle = candidate.get("teacher_overlap_oracle")
            if not isinstance(oracle, dict):
                oracle = {}
            overlap = oracle.get("vacancy_pair_overlap")
            if not isinstance(overlap, dict):
                overlap = {}
            records.append(
                {
                    "source_name": source_name,
                    "segment_index": int(segment.get("index", seg_index)),
                    "candidate_index": int(cand_index),
                    "fold_key": int(segment.get("index", seg_index)),
                    "segment_k": int(candidate.get("segment_k", diagnostic.get("segment_k", 0))),
                    "selected_by_planner": bool(diagnostic.get("selected_by_planner", False)),
                    "site_f1": _as_float(diagnostic.get("site_f1", oracle.get("f1", 0.0))),
                    "teacher_reward_sum": _as_float(diagnostic.get("teacher_reward_sum", 0.0)),
                    "vacancy_pair_precision": _as_float(
                        diagnostic.get("vacancy_pair_precision", overlap.get("precision", 0.0))
                    ),
                    "vacancy_pair_recall": _as_float(
                        diagnostic.get("vacancy_pair_recall", overlap.get("recall", 0.0))
                    ),
                    "vacancy_pair_f1": _as_float(
                        diagnostic.get("vacancy_pair_f1", overlap.get("f1", 0.0))
                    ),
                    "pairs": pairs,
                }
            )
    return records


def _fit_ridge(records: list[dict[str, Any]], l2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[float] = []
    for record in records:
        for item in record.get("pairs", []):
            if not isinstance(item, dict):
                continue
            features.append(list(item.get("_features", [])))
            labels.append(1.0 if bool(item.get("is_true_pair", False)) else -1.0)
    if not features:
        raise ValueError("No pair features found for calibration")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    x = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    pos = np.maximum((y > 0).sum(), 1)
    neg = np.maximum((y < 0).sum(), 1)
    weights = np.where(y > 0, float(neg / pos), 1.0)
    xw = x * weights[:, None]
    xtx = x.T @ xw
    reg = np.eye(xtx.shape[0], dtype=np.float64) * float(l2)
    reg[0, 0] = 0.0
    xty = x.T @ (weights * y)
    try:
        coef = np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(xtx + reg, xty, rcond=None)[0]
    return coef, mean, std


def _score_pair(item: dict[str, Any], score_name: str) -> float:
    return _as_float(item.get(score_name, 0.0))


def _apply_calibrated_scores(
    records: list[dict[str, Any]],
    *,
    score_name: str,
    l2: float,
) -> None:
    folds = sorted({int(record.get("fold_key", 0)) for record in records})
    if len(folds) < 2:
        coef, mean, std = _fit_ridge(records, l2)
        for record in records:
            for item in record.get("pairs", []):
                x = np.asarray(item.get("_features", []), dtype=np.float64)
                item[score_name] = float(np.concatenate([[1.0], (x - mean) / std]) @ coef)
        return
    for fold in folds:
        train = [record for record in records if int(record.get("fold_key", 0)) != fold]
        test = [record for record in records if int(record.get("fold_key", 0)) == fold]
        if not train or not test:
            continue
        coef, mean, std = _fit_ridge(train, l2)
        for record in test:
            for item in record.get("pairs", []):
                x = np.asarray(item.get("_features", []), dtype=np.float64)
                item[score_name] = float(np.concatenate([[1.0], (x - mean) / std]) @ coef)


def _rank_summary(records: list[dict[str, Any]], score_name: str) -> dict[str, Any]:
    rank_values: list[float] = []
    percentile_values: list[float] = []
    mrr_values: list[float] = []
    per_candidate_recall: dict[int, list[float]] = {k: [] for k in RANK_KEYS}
    per_candidate_fp: dict[int, list[float]] = {k: [] for k in RANK_KEYS}
    top128_true_counts: list[float] = []
    top128_precision: list[float] = []
    top128_recall: list[float] = []
    top128_f1: list[float] = []
    candidate_with_true = 0

    for record in records:
        pairs = [item for item in record.get("pairs", []) if isinstance(item, dict)]
        if not pairs:
            continue
        ranked = sorted(pairs, key=lambda item: _score_pair(item, score_name), reverse=True)
        true_ranks: list[int] = []
        for rank, item in enumerate(ranked, start=1):
            if bool(item.get("is_true_pair", False)):
                true_ranks.append(rank)
        true_total = len(true_ranks)
        if true_total:
            candidate_with_true += 1
            rank_values.extend(float(rank) for rank in true_ranks)
            percentile_values.extend(float(rank) / max(len(ranked), 1) for rank in true_ranks)
            mrr_values.extend(1.0 / float(rank) for rank in true_ranks)
        for k in RANK_KEYS:
            top = ranked[:k]
            true_count = sum(1 for item in top if bool(item.get("is_true_pair", False)))
            per_candidate_recall[k].append(float(true_count / max(true_total, 1)))
            per_candidate_fp[k].append(float((len(top) - true_count) / max(len(top), 1)))
        top128 = ranked[:128]
        true128 = sum(1 for item in top128 if bool(item.get("is_true_pair", False)))
        precision = float(true128 / max(len(top128), 1))
        recall = float(true128 / max(true_total, 1))
        f1 = 0.0 if precision + recall <= 0 else float(2.0 * precision * recall / (precision + recall))
        top128_true_counts.append(float(true128))
        top128_precision.append(precision)
        top128_recall.append(recall)
        top128_f1.append(f1)

    return {
        "candidate_count": int(len(records)),
        "candidate_with_true_pair_count": int(candidate_with_true),
        "true_pair_rank_mean": _mean(rank_values),
        "true_pair_rank_percentile_mean": _mean(percentile_values),
        "true_pair_mrr": _mean(mrr_values),
        "recall_at_rank": {str(k): _mean(per_candidate_recall[k]) for k in RANK_KEYS},
        "topk_false_positive_rate": {str(k): _mean(per_candidate_fp[k]) for k in RANK_KEYS},
        "top128_true_pair_count_mean": _mean(top128_true_counts),
        "top128_pair_precision": _mean(top128_precision),
        "top128_pair_recall": _mean(top128_recall),
        "top128_pair_f1": _mean(top128_f1),
    }


def _conditional_rank_items(
    record: dict[str, Any],
    condition: str,
    score_name: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    key_name = "_source_key" if condition == "source_to_destination" else "_destination_key"
    for item in record.get("pairs", []):
        key = item.get(key_name)
        if isinstance(key, tuple):
            groups[key].append(item)
    out: list[dict[str, Any]] = []
    for item in record.get("pairs", []):
        if not bool(item.get("is_true_pair", False)):
            continue
        group_key = item.get(key_name)
        group = groups.get(group_key, [])
        ranked = sorted(group, key=lambda entry: _score_pair(entry, score_name), reverse=True)
        true_source = item.get("_source_key")
        true_dest = item.get("_destination_key")
        rank = 0
        for idx, entry in enumerate(ranked, start=1):
            if entry.get("_source_key") == true_source and entry.get("_destination_key") == true_dest:
                rank = idx
                break
        if rank <= 0:
            continue
        label_counts: dict[str, int] = {label: 0 for label in TOP_LABEL_KEYS}
        for top_item in ranked[: min(32, len(ranked))]:
            label = str(top_item.get("pair_label", "false_pair"))
            label_counts[label] = label_counts.get(label, 0) + 1
        out.append(
            {
                "rank": float(rank),
                "group_size": float(len(ranked)),
                "rank_percentile": float(rank / max(len(ranked), 1)),
                "top32_label_counts": label_counts,
            }
        )
    return out


def _summarize_conditional_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"true_pair_count": 0}
    ranks = [_as_float(item.get("rank", 0.0)) for item in items]
    percentiles = [_as_float(item.get("rank_percentile", 0.0)) for item in items]
    group_sizes = [_as_float(item.get("group_size", 0.0)) for item in items]
    label_means: dict[str, float] = {}
    for label in TOP_LABEL_KEYS:
        label_means[label] = _mean([
            _as_float((item.get("top32_label_counts") or {}).get(label, 0.0))
            for item in items
            if isinstance(item.get("top32_label_counts"), dict)
        ])
    return {
        "true_pair_count": int(len(items)),
        "rank_mean": _mean(ranks),
        "rank_percentile_mean": _mean(percentiles),
        "mrr": _mean([1.0 / max(rank, 1.0) for rank in ranks]),
        "group_size_mean": _mean(group_sizes),
        "recall_at_rank": {
            str(k): _mean([1.0 if rank <= k else 0.0 for rank in ranks])
            for k in RANK_KEYS
        },
        "top32_label_count_mean": label_means,
    }


def _conditional_summary(records: list[dict[str, Any]], score_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition in ("source_to_destination", "destination_to_source"):
        items: list[dict[str, Any]] = []
        for record in records:
            items.extend(_conditional_rank_items(record, condition, score_name))
        out[condition] = _summarize_conditional_items(items)
    return out


def _group_summary(records: list[dict[str, Any]], score_names: list[str]) -> dict[str, Any]:
    if not records:
        return {"candidate_count": 0}
    return {
        "candidate_count": int(len(records)),
        "avg_site_f1": _mean([_as_float(record.get("site_f1", 0.0)) for record in records]),
        "avg_teacher_reward_sum": _mean([_as_float(record.get("teacher_reward_sum", 0.0)) for record in records]),
        "avg_vacancy_pair_precision": _mean([
            _as_float(record.get("vacancy_pair_precision", 0.0)) for record in records
        ]),
        "avg_vacancy_pair_recall": _mean([
            _as_float(record.get("vacancy_pair_recall", 0.0)) for record in records
        ]),
        "avg_vacancy_pair_f1": _mean([_as_float(record.get("vacancy_pair_f1", 0.0)) for record in records]),
        "score_summaries": {
            score_name: {
                "global_rank": _rank_summary(records, score_name),
                "conditional_rank": _conditional_summary(records, score_name),
            }
            for score_name in score_names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", action="append", required=True, help="v111 eval JSON. Can repeat.")
    parser.add_argument("--name", action="append", default=[], help="Optional name matching --eval-json.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    args = parser.parse_args()

    names = list(args.name)
    while len(names) < len(args.eval_json):
        names.append(Path(args.eval_json[len(names)]).stem)

    by_input: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for name, raw_path in zip(names, args.eval_json):
        path = Path(raw_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        records = _candidate_records(data, name)
        _apply_calibrated_scores(records, score_name="calibrated_interaction_score", l2=args.ridge_l2)
        all_records.extend(records)
        selected = [record for record in records if bool(record.get("selected_by_planner", False))]
        by_input[name] = {
            "eval_file": str(path),
            "completed_rollout_segments": data.get("completed_rollout_segments"),
            "stop_reason": data.get("stop_reason"),
            "chosen_k_histogram": data.get("chosen_k_histogram"),
            "cumulative": data.get("cumulative"),
            "tau_expected": data.get("tau_expected"),
            "selected_by_planner": _group_summary(
                selected,
                ["score", "destination_score", "moving_type_score", "calibrated_interaction_score"],
            ),
            "all_candidates": _group_summary(
                records,
                ["score", "destination_score", "moving_type_score", "calibrated_interaction_score"],
            ),
        }

    _apply_calibrated_scores(all_records, score_name="calibrated_interaction_score", l2=args.ridge_l2)
    selected_all = [record for record in all_records if bool(record.get("selected_by_planner", False))]
    summary = {
        "mode": "read-only v113 conditional compatibility calibration",
        "ridge_l2": float(args.ridge_l2),
        "inputs": by_input,
        "combined": {
            "selected_by_planner": _group_summary(
                selected_all,
                ["score", "destination_score", "moving_type_score", "calibrated_interaction_score"],
            ),
            "all_candidates": _group_summary(
                all_records,
                ["score", "destination_score", "moving_type_score", "calibrated_interaction_score"],
            ),
        },
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
