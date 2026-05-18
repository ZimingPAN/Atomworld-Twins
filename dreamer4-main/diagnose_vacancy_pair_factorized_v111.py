#!/usr/bin/env python3
"""Summarize v111 factorized vacancy-pair rank diagnostics.

The eval path writes per-pair source/destination/interaction/energy factors
only when --planner_vacancy_pair_factorized_diagnostic is enabled. This script
keeps the analysis read-only: it asks whether any model-visible factor can move
true terminal vacancy pairs into a usable top-k list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCORE_FIELDS = [
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
RANK_KEYS = [8, 16, 32, 64, 128, 256, 512, 1024]
LABELS = [
    "true_pair",
    "same_source_wrong_destination",
    "same_destination_wrong_source",
    "source_destination_unpaired",
    "false_pair",
]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _candidate_records(data: dict[str, Any]) -> list[dict[str, Any]]:
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
            pairs = projection.get("factorized_pair_scores")
            if not isinstance(pairs, list) or not pairs:
                continue
            diagnostic = candidate.get("candidate_joint_diagnostic")
            if not isinstance(diagnostic, dict):
                diagnostic = {}
            oracle = candidate.get("teacher_overlap_oracle")
            if not isinstance(oracle, dict):
                oracle = {}
            pair_overlap = oracle.get("vacancy_pair_overlap")
            if not isinstance(pair_overlap, dict):
                pair_overlap = {}
            rank = oracle.get("vacancy_pair_rank")
            if not isinstance(rank, dict):
                rank = {}
            records.append(
                {
                    "segment_index": int(segment.get("index", seg_index)),
                    "candidate_index": int(cand_index),
                    "segment_k": int(candidate.get("segment_k", diagnostic.get("segment_k", 0))),
                    "selected_by_planner": bool(diagnostic.get("selected_by_planner", False)),
                    "site_f1": _as_float(diagnostic.get("site_f1", oracle.get("f1", 0.0))),
                    "teacher_reward_sum": _as_float(diagnostic.get("teacher_reward_sum", 0.0)),
                    "vacancy_pair_f1": _as_float(
                        diagnostic.get("vacancy_pair_f1", pair_overlap.get("f1", 0.0))
                    ),
                    "vacancy_pair_precision": _as_float(
                        diagnostic.get("vacancy_pair_precision", pair_overlap.get("precision", 0.0))
                    ),
                    "vacancy_pair_recall": _as_float(
                        diagnostic.get("vacancy_pair_recall", pair_overlap.get("recall", 0.0))
                    ),
                    "rank": rank,
                    "pairs": [item for item in pairs if isinstance(item, dict)],
                }
            )
    return records


def _rank_stats_for_field(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    per_candidate_recall: dict[int, list[float]] = {k: [] for k in RANK_KEYS}
    per_candidate_false_rate: dict[int, list[float]] = {k: [] for k in RANK_KEYS}
    rank_values: list[float] = []
    percentile_values: list[float] = []
    mrr_values: list[float] = []
    true_score_values: list[float] = []
    false_score_values: list[float] = []
    found_candidate_count = 0

    for record in records:
        pairs = record.get("pairs", [])
        if not isinstance(pairs, list) or not pairs:
            continue
        ranked = sorted(pairs, key=lambda item: _as_float(item.get(field, 0.0)), reverse=True)
        true_ranks: list[int] = []
        for rank, item in enumerate(ranked, start=1):
            score = _as_float(item.get(field, 0.0))
            if bool(item.get("is_true_pair", False)):
                true_ranks.append(rank)
                true_score_values.append(score)
            else:
                false_score_values.append(score)
        if true_ranks:
            found_candidate_count += 1
            rank_values.extend(float(rank) for rank in true_ranks)
            percentile_values.extend(float(rank) / max(len(ranked), 1) for rank in true_ranks)
            mrr_values.extend(1.0 / float(rank) for rank in true_ranks)
        for k in RANK_KEYS:
            top = ranked[:k]
            true_count = sum(1 for item in top if bool(item.get("is_true_pair", False)))
            per_candidate_recall[k].append(float(true_count / max(len(true_ranks), 1)))
            per_candidate_false_rate[k].append(float((len(top) - true_count) / max(len(top), 1)))

    return {
        "candidate_count": int(len(records)),
        "candidate_with_true_pair_count": int(found_candidate_count),
        "true_pair_rank_mean": _mean(rank_values),
        "true_pair_rank_median": float(sorted(rank_values)[len(rank_values) // 2]) if rank_values else 0.0,
        "true_pair_rank_percentile_mean": _mean(percentile_values),
        "true_pair_mrr": _mean(mrr_values),
        "true_pair_score_mean": _mean(true_score_values),
        "false_pair_score_mean": _mean(false_score_values),
        "recall_at_rank": {str(k): _mean(per_candidate_recall[k]) for k in RANK_KEYS},
        "topk_false_positive_rate": {str(k): _mean(per_candidate_false_rate[k]) for k in RANK_KEYS},
    }


def _label_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    for record in records:
        pairs = record.get("pairs", [])
        if not isinstance(pairs, list):
            continue
        for item in pairs:
            label = str(item.get("pair_label", "false_pair"))
            by_label.setdefault(label, []).append(item)
    summary: dict[str, Any] = {}
    for label, items in by_label.items():
        field_summary = {
            field: _mean([_as_float(item.get(field, 0.0)) for item in items])
            for field in SCORE_FIELDS
        }
        summary[label] = {
            "count": int(len(items)),
            "scores": field_summary,
        }
    return summary


def _group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"candidate_count": 0}
    factor_summaries = {field: _rank_stats_for_field(records, field) for field in SCORE_FIELDS}
    best_recall128 = max(
        factor_summaries.items(),
        key=lambda item: _as_float(item[1].get("recall_at_rank", {}).get("128", 0.0)),
    )[0]
    best_percentile = min(
        factor_summaries.items(),
        key=lambda item: _as_float(item[1].get("true_pair_rank_percentile_mean", 1.0), 1.0),
    )[0]
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
        "best_factor_by_recall_at_128": best_recall128,
        "best_factor_by_rank_percentile": best_percentile,
        "factor_rank_summaries": factor_summaries,
        "label_distribution": _label_distribution(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    eval_path = Path(args.eval_json)
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    records = _candidate_records(data)
    selected = [record for record in records if bool(record.get("selected_by_planner", False))]
    unselected = [record for record in records if not bool(record.get("selected_by_planner", False))]
    summary = {
        "eval_file": str(eval_path),
        "completed_rollout_segments": data.get("completed_rollout_segments"),
        "requested_rollout_segments": data.get("requested_rollout_segments"),
        "stop_reason": data.get("stop_reason"),
        "chosen_k_histogram": data.get("chosen_k_histogram"),
        "cumulative": data.get("cumulative"),
        "tau_expected": data.get("tau_expected"),
        "all_candidates": _group_summary(records),
        "selected_by_planner": _group_summary(selected),
        "unselected_candidates": _group_summary(unselected),
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
