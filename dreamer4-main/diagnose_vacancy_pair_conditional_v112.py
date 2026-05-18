#!/usr/bin/env python3
"""Summarize v112 conditional vacancy-pair rank diagnostics.

This read-only diagnostic consumes v111 factorized pair payloads. It asks a
narrow question: if the true source is known, can any score rank the true
destination highly, and if the true destination is known, can any score rank the
true source highly? If not, the next model change needs a conditional
source-destination compatibility target rather than more top-k or scalar-head
tuning.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
RANK_KEYS = [1, 2, 4, 8, 16, 32, 64, 128]
TOP_LABEL_KEYS = [
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


def _position_key(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return tuple(int(v) for v in value)
    except (TypeError, ValueError):
        return None


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
                destination = _position_key(item.get("destination_position"))
                if source is None or destination is None:
                    continue
                pairs.append({**item, "_source_key": source, "_destination_key": destination})
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


def _conditional_rank_items(record: dict[str, Any], condition: str, field: str) -> list[dict[str, Any]]:
    pairs = record.get("pairs", [])
    if not isinstance(pairs, list):
        return []
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    key_name = "_source_key" if condition == "source_to_destination" else "_destination_key"
    for item in pairs:
        key = item.get(key_name)
        if isinstance(key, tuple):
            groups[key].append(item)
    out: list[dict[str, Any]] = []
    for item in pairs:
        if not bool(item.get("is_true_pair", False)):
            continue
        group_key = item.get(key_name)
        group = groups.get(group_key, [])
        if not group:
            continue
        ranked = sorted(group, key=lambda entry: _as_float(entry.get(field, 0.0)), reverse=True)
        true_source = item.get("_source_key")
        true_destination = item.get("_destination_key")
        rank = 0
        for idx, entry in enumerate(ranked, start=1):
            if entry.get("_source_key") == true_source and entry.get("_destination_key") == true_destination:
                rank = idx
                break
        if rank <= 0:
            continue
        label_counts: dict[str, int] = {key: 0 for key in TOP_LABEL_KEYS}
        for top_item in ranked[: min(32, len(ranked))]:
            label = str(top_item.get("pair_label", "false_pair"))
            label_counts[label] = label_counts.get(label, 0) + 1
        out.append(
            {
                "rank": float(rank),
                "group_size": float(len(ranked)),
                "rank_percentile": float(rank / max(len(ranked), 1)),
                "score": _as_float(item.get(field, 0.0)),
                "top32_label_counts": label_counts,
            }
        )
    return out


def _summarize_rank_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"true_pair_count": 0}
    ranks = [_as_float(item.get("rank", 0.0)) for item in items]
    group_sizes = [_as_float(item.get("group_size", 0.0)) for item in items]
    percentiles = [_as_float(item.get("rank_percentile", 0.0)) for item in items]
    scores = [_as_float(item.get("score", 0.0)) for item in items]
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
        "rank_median": float(sorted(ranks)[len(ranks) // 2]),
        "rank_percentile_mean": _mean(percentiles),
        "mrr": _mean([1.0 / max(rank, 1.0) for rank in ranks]),
        "group_size_mean": _mean(group_sizes),
        "true_score_mean": _mean(scores),
        "recall_at_rank": {
            str(k): _mean([1.0 if rank <= k else 0.0 for rank in ranks])
            for k in RANK_KEYS
        },
        "top32_label_count_mean": label_means,
    }


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"candidate_count": 0}
    condition_summary: dict[str, dict[str, Any]] = {}
    for condition in ("source_to_destination", "destination_to_source"):
        condition_summary[condition] = {}
        for field in SCORE_FIELDS:
            items: list[dict[str, Any]] = []
            for record in records:
                items.extend(_conditional_rank_items(record, condition, field))
            condition_summary[condition][field] = _summarize_rank_items(items)
    best: dict[str, str] = {}
    for condition, field_payloads in condition_summary.items():
        best[condition] = max(
            field_payloads.items(),
            key=lambda item: _as_float(item[1].get("recall_at_rank", {}).get("8", 0.0)),
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
        "best_field_by_recall_at_8": best,
        "conditional_rank": condition_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", action="append", required=True,
                        help="v111 eval JSON. Can be repeated.")
    parser.add_argument("--name", action="append", default=[],
                        help="Optional name for the matching --eval-json. Can be repeated.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    names = list(args.name)
    while len(names) < len(args.eval_json):
        names.append(Path(args.eval_json[len(names)]).stem)
    all_records: list[dict[str, Any]] = []
    by_input: dict[str, Any] = {}
    for name, raw_path in zip(names, args.eval_json):
        path = Path(raw_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        records = _candidate_records(data, name)
        all_records.extend(records)
        selected = [record for record in records if bool(record.get("selected_by_planner", False))]
        by_input[name] = {
            "eval_file": str(path),
            "completed_rollout_segments": data.get("completed_rollout_segments"),
            "stop_reason": data.get("stop_reason"),
            "chosen_k_histogram": data.get("chosen_k_histogram"),
            "cumulative": data.get("cumulative"),
            "tau_expected": data.get("tau_expected"),
            "all_candidates": _summarize_group(records),
            "selected_by_planner": _summarize_group(selected),
        }
    selected_all = [record for record in all_records if bool(record.get("selected_by_planner", False))]
    summary = {
        "inputs": by_input,
        "combined": {
            "all_candidates": _summarize_group(all_records),
            "selected_by_planner": _summarize_group(selected_all),
        },
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
