#!/usr/bin/env python3
"""Read-only v115 pair-compatibility distillation sample builder.

This script consumes v111/v113-style factorized vacancy-pair payloads and
rebuilds the leave-one-segment-out calibrated interaction score from v113.
It exports pair-level and candidate-level JSONL samples that can be used by a
future model-level distillation target, and it evaluates simple support-count
policies before any checkpoint training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import diagnose_vacancy_pair_interaction_v113 as v113


TOPK_BUDGETS = [4, 8, 16, 32, 64, 96, 128, 192, 256, 512]
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
    "calibrated_interaction_score",
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


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _position_list(value: Any) -> list[int] | None:
    if not isinstance(value, tuple) or len(value) != 3:
        return None
    return [int(value[0]), int(value[1]), int(value[2])]


def _pair_label(item: dict[str, Any]) -> str:
    if bool(item.get("is_true_pair", False)):
        return "true_pair"
    if bool(item.get("same_source_wrong_destination", False)):
        return "same_source_wrong_destination"
    if bool(item.get("same_destination_wrong_source", False)):
        return "same_destination_wrong_source"
    if bool(item.get("source_destination_unpaired", False)):
        return "source_destination_unpaired"
    return str(item.get("pair_label", "false_pair"))


def _true_pair_count(record: dict[str, Any]) -> int:
    return sum(1 for item in record.get("pairs", []) if bool(item.get("is_true_pair", False)))


def _pair_set(items: list[dict[str, Any]]) -> set[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    out: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    for item in items:
        source = item.get("_source_key")
        dest = item.get("_destination_key")
        if isinstance(source, tuple) and isinstance(dest, tuple):
            out.add((source, dest))
    return out


def _endpoint_set(items: list[dict[str, Any]]) -> set[tuple[int, int, int]]:
    out: set[tuple[int, int, int]] = set()
    for item in items:
        source = item.get("_source_key")
        dest = item.get("_destination_key")
        if isinstance(source, tuple):
            out.add(source)
        if isinstance(dest, tuple):
            out.add(dest)
    return out


def _support_metrics(selected: list[dict[str, Any]], true_items: list[dict[str, Any]]) -> dict[str, float]:
    selected_pairs = _pair_set(selected)
    true_pairs = _pair_set(true_items)
    pair_tp = len(selected_pairs & true_pairs)
    pair_precision = float(pair_tp / max(len(selected_pairs), 1))
    pair_recall = float(pair_tp / max(len(true_pairs), 1))

    selected_sites = _endpoint_set(selected)
    true_sites = _endpoint_set(true_items)
    site_tp = len(selected_sites & true_sites)
    site_precision = float(site_tp / max(len(selected_sites), 1))
    site_recall = float(site_tp / max(len(true_sites), 1))
    return {
        "selected_pair_count": float(len(selected_pairs)),
        "true_pair_count": float(len(true_pairs)),
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pair_f1": _f1(pair_precision, pair_recall),
        "selected_endpoint_count": float(len(selected_sites)),
        "true_endpoint_count": float(len(true_sites)),
        "endpoint_precision": site_precision,
        "endpoint_recall": site_recall,
        "endpoint_f1": _f1(site_precision, site_recall),
    }


def _topk_metrics(record: dict[str, Any], score_name: str, k: int) -> dict[str, float]:
    pairs = [item for item in record.get("pairs", []) if isinstance(item, dict)]
    ranked = sorted(pairs, key=lambda item: _as_float(item.get(score_name, 0.0)), reverse=True)
    true_items = [item for item in pairs if bool(item.get("is_true_pair", False))]
    return _support_metrics(ranked[: max(int(k), 0)], true_items)


def _threshold_metrics(record: dict[str, Any], score_name: str, threshold: float) -> dict[str, float]:
    pairs = [item for item in record.get("pairs", []) if isinstance(item, dict)]
    selected = [item for item in pairs if _as_float(item.get(score_name, 0.0)) > float(threshold)]
    if not selected and pairs:
        selected = [max(pairs, key=lambda item: _as_float(item.get(score_name, 0.0)))]
    true_items = [item for item in pairs if bool(item.get("is_true_pair", False))]
    return _support_metrics(selected, true_items)


def _best_budget(record: dict[str, Any], score_name: str) -> dict[str, float]:
    best: dict[str, float] | None = None
    for budget in TOPK_BUDGETS:
        metrics = _topk_metrics(record, score_name, budget)
        metrics["budget"] = float(budget)
        if best is None or (
            metrics["pair_f1"],
            metrics["endpoint_f1"],
            metrics["pair_recall"],
            -metrics["selected_pair_count"],
        ) > (
            best["pair_f1"],
            best["endpoint_f1"],
            best["pair_recall"],
            -best["selected_pair_count"],
        ):
            best = metrics
    return best or {}


def _score_quantiles(pairs: list[dict[str, Any]], score_name: str) -> dict[str, float]:
    values = sorted(_as_float(item.get(score_name, 0.0)) for item in pairs)
    if not values:
        return {}
    def pick(q: float) -> float:
        idx = min(max(int(round(q * (len(values) - 1))), 0), len(values) - 1)
        return float(values[idx])
    return {"p50": pick(0.50), "p75": pick(0.75), "p90": pick(0.90), "p95": pick(0.95), "p99": pick(0.99)}


def _candidate_sample(record: dict[str, Any], score_name: str) -> dict[str, Any]:
    pairs = [item for item in record.get("pairs", []) if isinstance(item, dict)]
    true_count = _true_pair_count(record)
    fixed = {f"top{k}": _topk_metrics(record, score_name, k) for k in TOPK_BUDGETS}
    threshold0 = _threshold_metrics(record, score_name, 0.0)
    top_true_count = _topk_metrics(record, score_name, max(true_count, 1))
    best = _best_budget(record, score_name)
    return {
        "source_name": record.get("source_name"),
        "segment_index": int(record.get("segment_index", 0)),
        "candidate_index": int(record.get("candidate_index", 0)),
        "fold_key": int(record.get("fold_key", 0)),
        "segment_k": int(record.get("segment_k", 0)),
        "selected_by_planner": bool(record.get("selected_by_planner", False)),
        "site_f1": _as_float(record.get("site_f1", 0.0)),
        "teacher_reward_sum": _as_float(record.get("teacher_reward_sum", 0.0)),
        "vacancy_pair_precision": _as_float(record.get("vacancy_pair_precision", 0.0)),
        "vacancy_pair_recall": _as_float(record.get("vacancy_pair_recall", 0.0)),
        "vacancy_pair_f1": _as_float(record.get("vacancy_pair_f1", 0.0)),
        "true_pair_count": int(true_count),
        "pair_count": int(len(pairs)),
        "calibrated_score_quantiles": _score_quantiles(pairs, score_name),
        "policies": {
            **fixed,
            "threshold_gt_0": threshold0,
            "top_true_pair_count_oracle": top_true_count,
            "best_fixed_budget_oracle": best,
        },
        "support_count_targets": {
            "true_pair_count": int(true_count),
            "threshold_gt_0_count": int(threshold0.get("selected_pair_count", 0)),
            "best_fixed_budget": int(best.get("budget", 0)),
            "best_fixed_budget_pair_f1": _as_float(best.get("pair_f1", 0.0)),
        },
    }


def _pair_sample(record: dict[str, Any], item: dict[str, Any], rank: int) -> dict[str, Any]:
    source = _position_list(item.get("_source_key"))
    dest = _position_list(item.get("_destination_key"))
    out = {
        "source_name": record.get("source_name"),
        "segment_index": int(record.get("segment_index", 0)),
        "candidate_index": int(record.get("candidate_index", 0)),
        "fold_key": int(record.get("fold_key", 0)),
        "segment_k": int(record.get("segment_k", 0)),
        "selected_by_planner": bool(record.get("selected_by_planner", False)),
        "source_position": source,
        "destination_position": dest,
        "label": 1 if bool(item.get("is_true_pair", False)) else 0,
        "pair_label": _pair_label(item),
        "calibrated_rank": int(rank),
    }
    for field in SCORE_FIELDS:
        out[field] = _as_float(item.get(field, 0.0))
    features = item.get("_features", [])
    if isinstance(features, list):
        out["features"] = [_as_float(value) for value in features]
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _summarize_policy(candidate_rows: list[dict[str, Any]], policy_name: str) -> dict[str, float]:
    rows = [
        (row.get("policies") or {}).get(policy_name, {})
        for row in candidate_rows
        if isinstance((row.get("policies") or {}).get(policy_name, {}), dict)
    ]
    return {
        "candidate_count": float(len(rows)),
        "selected_pair_count": _mean([_as_float(row.get("selected_pair_count", 0.0)) for row in rows]),
        "pair_precision": _mean([_as_float(row.get("pair_precision", 0.0)) for row in rows]),
        "pair_recall": _mean([_as_float(row.get("pair_recall", 0.0)) for row in rows]),
        "pair_f1": _mean([_as_float(row.get("pair_f1", 0.0)) for row in rows]),
        "endpoint_precision": _mean([_as_float(row.get("endpoint_precision", 0.0)) for row in rows]),
        "endpoint_recall": _mean([_as_float(row.get("endpoint_recall", 0.0)) for row in rows]),
        "endpoint_f1": _mean([_as_float(row.get("endpoint_f1", 0.0)) for row in rows]),
    }


def _summarize_candidates(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_rows:
        return {"candidate_count": 0}
    policy_names = list((candidate_rows[0].get("policies") or {}).keys())
    selected = [row for row in candidate_rows if bool(row.get("selected_by_planner", False))]
    return {
        "candidate_count": int(len(candidate_rows)),
        "selected_candidate_count": int(len(selected)),
        "avg_true_pair_count": _mean([_as_float(row.get("true_pair_count", 0.0)) for row in candidate_rows]),
        "avg_site_f1_label": _mean([_as_float(row.get("site_f1", 0.0)) for row in candidate_rows]),
        "avg_teacher_reward_sum": _mean([_as_float(row.get("teacher_reward_sum", 0.0)) for row in candidate_rows]),
        "avg_original_vacancy_pair_f1": _mean([
            _as_float(row.get("vacancy_pair_f1", 0.0)) for row in candidate_rows
        ]),
        "policies_all_candidates": {
            name: _summarize_policy(candidate_rows, name)
            for name in policy_names
        },
        "policies_selected_by_planner": {
            name: _summarize_policy(selected, name)
            for name in policy_names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", action="append", required=True, help="v111 eval JSON. Can repeat.")
    parser.add_argument("--name", action="append", default=[], help="Optional name matching --eval-json.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--max-pairs-per-candidate-jsonl", type=int, default=0,
                        help="Limit exported pair rows per candidate after calibrated sorting; 0 means all.")
    args = parser.parse_args()

    names = list(args.name)
    while len(names) < len(args.eval_json):
        names.append(Path(args.eval_json[len(names)]).stem)

    records: list[dict[str, Any]] = []
    for name, raw_path in zip(names, args.eval_json):
        path = Path(raw_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        records.extend(v113._candidate_records(data, name))
    v113._apply_calibrated_scores(records, score_name="calibrated_interaction_score", l2=float(args.ridge_l2))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = [_candidate_sample(record, "calibrated_interaction_score") for record in records]
    pair_rows: list[dict[str, Any]] = []
    for record in records:
        ranked = sorted(
            [item for item in record.get("pairs", []) if isinstance(item, dict)],
            key=lambda item: _as_float(item.get("calibrated_interaction_score", 0.0)),
            reverse=True,
        )
        limit = int(args.max_pairs_per_candidate_jsonl)
        export_ranked = ranked if limit <= 0 else ranked[:limit]
        for rank, item in enumerate(export_ranked, start=1):
            pair_rows.append(_pair_sample(record, item, rank))

    _write_jsonl(out_dir / "pair_distillation_samples_v115.jsonl", pair_rows)
    _write_jsonl(out_dir / "candidate_support_count_samples_v115.jsonl", candidate_rows)

    summary = {
        "mode": "read-only v115 calibrated interaction distillation samples",
        "ridge_l2": float(args.ridge_l2),
        "eval_json": list(args.eval_json),
        "candidate_count": int(len(candidate_rows)),
        "pair_sample_count": int(len(pair_rows)),
        "max_pairs_per_candidate_jsonl": int(args.max_pairs_per_candidate_jsonl),
        "summary": _summarize_candidates(candidate_rows),
        "output_files": {
            "pair_samples": str(out_dir / "pair_distillation_samples_v115.jsonl"),
            "candidate_samples": str(out_dir / "candidate_support_count_samples_v115.jsonl"),
            "summary": str(out_dir / "stage_summary.json"),
        },
    }
    (out_dir / "stage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
