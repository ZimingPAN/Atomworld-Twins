#!/usr/bin/env python3
"""Summarize v107 vacancy-pair rank diagnostics from long-eval JSON.

This is read-only post-processing. It assumes eval_macro_long_trajectory.py was
run with --planner_vacancy_pair_rank_diagnostic and candidate joint diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _mean_field(records: list[dict[str, Any]], field: str) -> float:
    return _mean([_as_float(record.get(field, 0.0)) for record in records])


def _mean_nested_dict(records: list[dict[str, Any]], field: str, keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        vals: list[float] = []
        for record in records:
            nested = record.get(field)
            if isinstance(nested, dict):
                vals.append(_as_float(nested.get(key, 0.0)))
        out[key] = _mean(vals)
    return out


def _candidate_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for segment in data.get("segments", []):
        if not isinstance(segment, dict):
            continue
        for candidate in segment.get("planner_candidates", []):
            if not isinstance(candidate, dict):
                continue
            diagnostic = candidate.get("candidate_joint_diagnostic")
            if not isinstance(diagnostic, dict):
                diagnostic = {}
            oracle = candidate.get("teacher_overlap_oracle")
            if not isinstance(oracle, dict):
                oracle = {}
            rank = oracle.get("vacancy_pair_rank")
            if not isinstance(rank, dict):
                rank = {}
            overlap = oracle.get("vacancy_pair_overlap")
            if not isinstance(overlap, dict):
                overlap = {}
            records.append(
                {
                    **diagnostic,
                    "segment_k": int(candidate.get("segment_k", diagnostic.get("segment_k", 0))),
                    "selection_score": _as_float(candidate.get("selection_score", diagnostic.get("selection_score", 0.0))),
                    "selected_by_planner": bool(diagnostic.get("selected_by_planner", False)),
                    "rank": rank,
                    "pair_overlap": overlap,
                }
            )
    return records


def _rank_group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    rank_records = [record for record in records if record.get("rank")]
    if not rank_records:
        return {"count": 0}
    rank_payloads = [record["rank"] for record in rank_records if isinstance(record.get("rank"), dict)]
    rank_keys = ["8", "16", "32", "64", "128", "256", "512", "1024"]
    return {
        "count": int(len(rank_records)),
        "avg_site_f1": _mean_field(rank_records, "site_f1"),
        "avg_vacancy_pair_precision": _mean_field(rank_records, "vacancy_pair_precision"),
        "avg_vacancy_pair_recall": _mean_field(rank_records, "vacancy_pair_recall"),
        "avg_vacancy_pair_f1": _mean_field(rank_records, "vacancy_pair_f1"),
        "avg_teacher_reward_sum": _mean_field(rank_records, "teacher_reward_sum"),
        "avg_projected_changed_count": _mean_field(rank_records, "projected_changed_count"),
        "avg_vacancy_pair_selected_count": _mean_field(rank_records, "vacancy_pair_selected_count"),
        "avg_ranked_pair_count": _mean([_as_float(rank.get("ranked_pair_count", 0.0)) for rank in rank_payloads]),
        "avg_teacher_pair_count": _mean([_as_float(rank.get("teacher_pair_count", 0.0)) for rank in rank_payloads]),
        "avg_true_pair_found_recall": _mean([_as_float(rank.get("teacher_pair_found_recall", 0.0)) for rank in rank_payloads]),
        "avg_true_pair_rank_mean": _mean([_as_float(rank.get("teacher_pair_rank_mean", 0.0)) for rank in rank_payloads]),
        "avg_true_pair_rank_median": _mean([_as_float(rank.get("teacher_pair_rank_median", 0.0)) for rank in rank_payloads]),
        "avg_true_pair_rank_percentile_mean": _mean([
            _as_float(rank.get("teacher_pair_rank_percentile_mean", 0.0)) for rank in rank_payloads
        ]),
        "avg_true_pair_mrr": _mean([_as_float(rank.get("teacher_pair_mrr", 0.0)) for rank in rank_payloads]),
        "avg_true_pair_typed_rank_accuracy": _mean([
            _as_float(rank.get("teacher_pair_typed_rank_accuracy", 0.0)) for rank in rank_payloads
        ]),
        "avg_recall_at_rank": _mean_nested_dict(rank_payloads, "teacher_pair_recall_at_rank", rank_keys),
        "avg_topk_false_positive_rate": _mean_nested_dict(rank_payloads, "topk_false_positive_rate", rank_keys),
        "avg_topk_true_pair_count": _mean_nested_dict(rank_payloads, "topk_true_pair_count", rank_keys),
        "avg_topk_source_hard_negative_count": _mean_nested_dict(
            rank_payloads, "topk_source_hard_negative_count", rank_keys
        ),
        "avg_topk_destination_hard_negative_count": _mean_nested_dict(
            rank_payloads, "topk_destination_hard_negative_count", rank_keys
        ),
        "avg_topk_source_destination_unpaired_count": _mean_nested_dict(
            rank_payloads, "topk_source_destination_unpaired_count", rank_keys
        ),
        "avg_topk_type_mismatch_count": _mean_nested_dict(rank_payloads, "topk_type_mismatch_count", rank_keys),
        "avg_topk_true_score_mean": _mean_nested_dict(rank_payloads, "topk_true_score_mean", rank_keys),
        "avg_topk_false_score_mean": _mean_nested_dict(rank_payloads, "topk_false_score_mean", rank_keys),
    }


def _selector_summary(data: dict[str, Any]) -> dict[str, Any]:
    candidate_joint = data.get("candidate_joint_diagnostic")
    if not isinstance(candidate_joint, dict):
        return {}
    selected = candidate_joint.get("selected_by_planner", {})
    upper = candidate_joint.get("selector_upper_bounds", {})
    return {
        "selected_by_planner": selected if isinstance(selected, dict) else {},
        "selector_upper_bounds": upper if isinstance(upper, dict) else {},
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
        "candidate_count": int(len(records)),
        "selected_candidate_count": int(len(selected)),
        "all_candidates_rank": _rank_group_summary(records),
        "selected_by_planner_rank": _rank_group_summary(selected),
        "unselected_candidates_rank": _rank_group_summary(unselected),
        "candidate_joint_summary": _selector_summary(data),
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
