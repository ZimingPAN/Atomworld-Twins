#!/usr/bin/env python3
"""Read-only v136 diagnostic for v135 budget-to-projection retention.

This script only post-processes existing v135 long-eval JSON files.  It does
not import torch, does not train, and does not run a new closed-loop eval.

The v135 compact long-eval artifacts expose factorized pair scores for planner
candidates and the final selected pair overlap, but they do not expose the full
teacher terminal vacancy-pair target for every candidate.  Consequently, the
rank/retention metrics below are scoped to observed pairs that survived into
the selected projection and overlapped the teacher target.  This is enough to
test whether the applied budget cap discards already-observed true pairs, while
full true-pair recall still requires a non-compact rank diagnostic.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_ROOT = Path("results/natural_teacher_support32_sequence_rollout_0517")
DEFAULT_DIAGNOSTIC = (
    DEFAULT_ROOT
    / "v135_guarded_budget_projection_diagnostic_smoke20"
    / "eval_long_guarded_budget_projection_diagnostic_20.json"
)
DEFAULT_REPLACE = (
    DEFAULT_ROOT
    / "v135_guarded_budget_projection_replace_retry_smoke20"
    / "eval_long_guarded_budget_projection_replace_20.json"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "v136_pair_retention_readonly"
DEFAULT_BUDGETS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _hist(values: list[Any]) -> dict[str, int]:
    return {str(key): int(count) for key, count in sorted(Counter(values).items(), key=lambda item: str(item[0]))}


def _pair_key(record: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(_as_int(item) for item in record.get("source_position", [])),
        tuple(_as_int(item) for item in record.get("destination_position", [])),
    )


def _overlap_pair_key(record: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(_as_int(item) for item in record.get("source_position", record.get("source", []))),
        tuple(_as_int(item) for item in record.get("destination_position", record.get("destination", []))),
    )


def _selected_candidate(segment: dict[str, Any]) -> dict[str, Any]:
    candidates = [candidate for candidate in segment.get("planner_candidates", []) if isinstance(candidate, dict)]
    if not candidates:
        return {}
    segment_k = _as_int(segment.get("segment_k"))
    same_k = [candidate for candidate in candidates if _as_int(candidate.get("segment_k")) == segment_k]
    pool = same_k or candidates
    return max(pool, key=lambda candidate: _as_float(candidate.get("selection_score")))


def _selector_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    selector = candidate.get("planner_candidate_pareto_selector")
    return selector if isinstance(selector, dict) else {}


def _selector_prediction(selector: dict[str, Any], key: str) -> float:
    predictions = selector.get("predictions")
    if isinstance(predictions, dict) and key in predictions:
        return _as_float(predictions.get(key))
    return _as_float(selector.get(f"pred_{key}"))


def _score_by_key(scores: list[dict[str, Any]]) -> dict[tuple[tuple[int, ...], tuple[int, ...]], dict[str, Any]]:
    out: dict[tuple[tuple[int, ...], tuple[int, ...]], dict[str, Any]] = {}
    for score in scores:
        if not isinstance(score, dict):
            continue
        key = _pair_key(score)
        if key not in out:
            out[key] = score
    return out


def _summarize_numeric(rows: list[dict[str, Any]], key: str) -> float:
    return _mean([_as_float(row.get(key)) for row in rows])


def _summarize_numeric_where(rows: list[dict[str, Any]], key: str, guard_key: str) -> float:
    return _mean([_as_float(row.get(key)) for row in rows if _as_float(row.get(guard_key)) > 0.0])


def _retention_at_budgets(
    observed_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]],
    scores: list[dict[str, Any]],
    budgets: list[int],
) -> dict[str, float]:
    if not observed_pairs:
        return {str(budget): 0.0 for budget in budgets}
    out: dict[str, float] = {}
    ordered_keys = [_pair_key(score) for score in scores]
    for budget in budgets:
        top_keys = set(ordered_keys[: max(0, int(budget))])
        out[str(budget)] = float(len(observed_pairs & top_keys) / max(len(observed_pairs), 1))
    return out


def _retained_count(
    observed_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]],
    scores: list[dict[str, Any]],
    budget: int,
) -> int:
    if budget <= 0 or not observed_pairs:
        return 0
    return int(len(observed_pairs & set(_pair_key(score) for score in scores[:budget])))


def _hard_negative_counts(
    observed_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]],
    scores: list[dict[str, Any]],
    budget: int,
) -> dict[str, int]:
    if budget <= 0 or not scores:
        return {
            "same_source_wrong_destination": 0,
            "same_destination_wrong_source": 0,
            "source_destination_unpaired": 0,
            "observed_true_in_top_budget": 0,
            "false_pair_in_top_budget": 0,
        }
    true_sources = {source for source, _ in observed_pairs}
    true_destinations = {destination for _, destination in observed_pairs}
    counts = Counter()
    for score in scores[:budget]:
        key = _pair_key(score)
        if key in observed_pairs:
            counts["observed_true_in_top_budget"] += 1
            continue
        source, destination = key
        counts["false_pair_in_top_budget"] += 1
        if source in true_sources and destination in true_destinations:
            counts["source_destination_unpaired"] += 1
        elif source in true_sources:
            counts["same_source_wrong_destination"] += 1
        elif destination in true_destinations:
            counts["same_destination_wrong_source"] += 1
    return {
        "same_source_wrong_destination": int(counts["same_source_wrong_destination"]),
        "same_destination_wrong_source": int(counts["same_destination_wrong_source"]),
        "source_destination_unpaired": int(counts["source_destination_unpaired"]),
        "observed_true_in_top_budget": int(counts["observed_true_in_top_budget"]),
        "false_pair_in_top_budget": int(counts["false_pair_in_top_budget"]),
    }


def _top_false_score_mean(
    observed_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]],
    scores: list[dict[str, Any]],
    budget: int,
    field: str = "score",
) -> float:
    values = [_as_float(score.get(field)) for score in scores[:budget] if _pair_key(score) not in observed_pairs]
    return _mean(values)


def _analyze_stage(name: str, eval_path: Path, budgets: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for segment in data.get("segments", []):
        if not isinstance(segment, dict):
            continue
        candidate = _selected_candidate(segment)
        selector = _selector_payload(candidate)
        vpd = candidate.get("vacancy_pair_projection_diagnostic")
        if not isinstance(vpd, dict):
            vpd = {}
        scores = vpd.get("factorized_pair_scores")
        if not isinstance(scores, list):
            scores = []
        scores = sorted([score for score in scores if isinstance(score, dict)], key=lambda score: _as_int(score.get("rank"), 10**9))
        score_map = _score_by_key(scores)
        overlap = segment.get("vacancy_pair_overlap")
        if not isinstance(overlap, dict):
            overlap = {}
        observed_pairs = {
            _overlap_pair_key(pair)
            for pair in overlap.get("overlap_pairs", [])
            if isinstance(pair, dict)
        }
        observed_score_records = [score_map[key] for key in observed_pairs if key in score_map]
        observed_ranks = [_as_float(score.get("rank")) for score in observed_score_records]
        applied_budget = _as_int(
            selector.get("projection_rerun_pair_budget", selector.get("budget", vpd.get("selected_pair_count")))
        )
        selected_pair_count = _as_int(vpd.get("selected_pair_count", overlap.get("selected_pair_count", applied_budget)))
        retained_at_applied = _retained_count(observed_pairs, scores, selected_pair_count)
        top_false_score = _top_false_score_mean(observed_pairs, scores, selected_pair_count)
        observed_score_mean = _mean([_as_float(score.get("score")) for score in observed_score_records])
        hard_counts = _hard_negative_counts(observed_pairs, scores, selected_pair_count)
        teacher_pair_count = _as_int(overlap.get("teacher_pair_count"))
        actual_pair_recall = _as_float(overlap.get("recall"))
        row = {
            "stage": name,
            "segment_index": _as_int(segment.get("index", len(rows))),
            "segment_k": _as_int(segment.get("segment_k")),
            "candidate_segment_k": _as_int(candidate.get("segment_k")),
            "candidate_selection_score": _as_float(candidate.get("selection_score")),
            "selector_mode": str(selector.get("mode", "")),
            "selector_score": _as_float(selector.get("score")),
            "selector_budget": _as_int(selector.get("budget")),
            "selector_projection_rerun_budget": _as_int(selector.get("projection_rerun_pair_budget")),
            "selector_pred_teacher_reward_sum": _selector_prediction(selector, "teacher_reward_sum"),
            "selector_pred_site_f1": _selector_prediction(selector, "site_f1"),
            "selector_pred_pair_precision": _selector_prediction(selector, "pair_precision"),
            "selector_pred_pair_recall": _selector_prediction(selector, "pair_recall"),
            "selector_pred_pair_f1": _selector_prediction(selector, "pair_f1"),
            "selector_pred_endpoint_f1": _selector_prediction(selector, "endpoint_f1"),
            "factorized_pair_score_count": int(len(scores)),
            "selected_pair_count": selected_pair_count,
            "observed_overlap_pair_count": int(len(observed_pairs)),
            "teacher_pair_count": teacher_pair_count,
            "actual_pair_precision": _as_float(overlap.get("precision")),
            "actual_pair_recall": actual_pair_recall,
            "actual_pair_f1": _as_float(overlap.get("f1")),
            "actual_typed_endpoint_accuracy": _as_float(overlap.get("typed_endpoint_accuracy")),
            "actual_site_f1": _as_float((segment.get("proposal_overlap") or {}).get("f1")),
            "traditional_delta_e": _as_float(segment.get("traditional_kmc_delta_e")),
            "predicted_delta_e": _as_float(segment.get("predicted_delta_e")),
            "observed_pair_rank_mean": _mean(observed_ranks),
            "observed_pair_rank_median": _median(observed_ranks),
            "observed_pair_rank_percentile_mean": _mean(
                [rank / max(len(scores), 1) for rank in observed_ranks]
            ),
            "observed_pair_score_mean": observed_score_mean,
            "top_selected_false_score_mean": top_false_score,
            "observed_minus_false_score_mean": observed_score_mean - top_false_score,
            "observed_retained_at_selected_budget": retained_at_applied,
            "observed_retention_at_selected_budget": float(retained_at_applied / max(len(observed_pairs), 1)),
            "observed_retention_at_128": _retention_at_budgets(observed_pairs, scores, [128]).get("128", 0.0),
            "observed_retention_at_budget": _retention_at_budgets(observed_pairs, scores, budgets),
            "actual_pair_recall_minus_pred": actual_pair_recall - _selector_prediction(selector, "pair_recall"),
            "actual_pair_precision_minus_pred": _as_float(overlap.get("precision")) - _selector_prediction(selector, "pair_precision"),
            "actual_pair_f1_minus_pred": _as_float(overlap.get("f1")) - _selector_prediction(selector, "pair_f1"),
            **{f"top_budget_{key}": value for key, value in hard_counts.items()},
        }
        rows.append(row)

    eval_selector = data.get("planner_candidate_pareto_selector")
    if not isinstance(eval_selector, dict):
        eval_selector = {}
    cumulative = data.get("cumulative")
    if not isinstance(cumulative, dict):
        cumulative = {}
    tau_expected = data.get("tau_expected")
    if not isinstance(tau_expected, dict):
        tau_expected = {}
    retention_at_budget: dict[str, float] = {}
    for budget in budgets:
        retention_at_budget[str(budget)] = _mean(
            [
                _as_float(row.get("observed_retention_at_budget", {}).get(str(budget)))
                for row in rows
            ]
        )
    summary = {
        "eval_path": str(eval_path),
        "completed_rollout_segments": data.get("completed_rollout_segments"),
        "stop_reason": data.get("stop_reason"),
        "chosen_k_histogram": data.get("chosen_k_histogram"),
        "cumulative": cumulative,
        "tau_expected": tau_expected,
        "planner_candidate_pareto_selector": eval_selector,
        "segment_count": int(len(rows)),
        "selected_k_histogram": _hist([row["segment_k"] for row in rows]),
        "selector_budget_histogram": _hist([row["selector_budget"] for row in rows]),
        "selected_pair_count_histogram": _hist([row["selected_pair_count"] for row in rows]),
        "actual_site_f1_mean": _summarize_numeric(rows, "actual_site_f1"),
        "actual_pair_precision_mean": _summarize_numeric(rows, "actual_pair_precision"),
        "actual_pair_recall_mean": _summarize_numeric(rows, "actual_pair_recall"),
        "actual_pair_f1_mean": _summarize_numeric(rows, "actual_pair_f1"),
        "typed_endpoint_accuracy_mean": _summarize_numeric(rows, "actual_typed_endpoint_accuracy"),
        "selected_pair_count_mean": _summarize_numeric(rows, "selected_pair_count"),
        "teacher_pair_count_mean": _summarize_numeric(rows, "teacher_pair_count"),
        "observed_overlap_pair_count_mean": _summarize_numeric(rows, "observed_overlap_pair_count"),
        "observed_overlap_pair_count_total": int(sum(_as_int(row.get("observed_overlap_pair_count")) for row in rows)),
        "teacher_pair_count_total": int(sum(_as_int(row.get("teacher_pair_count")) for row in rows)),
        "segments_with_no_observed_overlap_pairs": int(sum(1 for row in rows if row["observed_overlap_pair_count"] <= 0)),
        "segments_with_observed_overlap_pairs": int(sum(1 for row in rows if row["observed_overlap_pair_count"] > 0)),
        "observed_pair_rank_mean": _summarize_numeric(rows, "observed_pair_rank_mean"),
        "observed_pair_rank_mean_on_hit_segments": _summarize_numeric_where(
            rows, "observed_pair_rank_mean", "observed_overlap_pair_count"
        ),
        "observed_pair_rank_median_mean": _summarize_numeric(rows, "observed_pair_rank_median"),
        "observed_pair_rank_median_on_hit_segments": _summarize_numeric_where(
            rows, "observed_pair_rank_median", "observed_overlap_pair_count"
        ),
        "observed_pair_rank_percentile_mean": _summarize_numeric(rows, "observed_pair_rank_percentile_mean"),
        "observed_pair_rank_percentile_on_hit_segments": _summarize_numeric_where(
            rows, "observed_pair_rank_percentile_mean", "observed_overlap_pair_count"
        ),
        "observed_pair_score_mean": _summarize_numeric(rows, "observed_pair_score_mean"),
        "observed_pair_score_mean_on_hit_segments": _summarize_numeric_where(
            rows, "observed_pair_score_mean", "observed_overlap_pair_count"
        ),
        "top_selected_false_score_mean": _summarize_numeric(rows, "top_selected_false_score_mean"),
        "observed_minus_false_score_mean": _summarize_numeric(rows, "observed_minus_false_score_mean"),
        "observed_minus_false_score_mean_on_hit_segments": _summarize_numeric_where(
            rows, "observed_minus_false_score_mean", "observed_overlap_pair_count"
        ),
        "observed_retention_at_selected_budget_mean": _summarize_numeric(rows, "observed_retention_at_selected_budget"),
        "observed_retention_at_selected_budget_on_hit_segments": _summarize_numeric_where(
            rows, "observed_retention_at_selected_budget", "observed_overlap_pair_count"
        ),
        "observed_retention_at_128_mean": _summarize_numeric(rows, "observed_retention_at_128"),
        "observed_retention_at_128_on_hit_segments": _summarize_numeric_where(
            rows, "observed_retention_at_128", "observed_overlap_pair_count"
        ),
        "observed_retention_at_budget_mean": retention_at_budget,
        "observed_true_in_top_budget_mean": _summarize_numeric(rows, "top_budget_observed_true_in_top_budget"),
        "false_pair_in_top_budget_mean": _summarize_numeric(rows, "top_budget_false_pair_in_top_budget"),
        "same_source_wrong_destination_mean": _summarize_numeric(rows, "top_budget_same_source_wrong_destination"),
        "same_destination_wrong_source_mean": _summarize_numeric(rows, "top_budget_same_destination_wrong_source"),
        "source_destination_unpaired_mean": _summarize_numeric(rows, "top_budget_source_destination_unpaired"),
        "selector_pred_pair_recall_mean": _summarize_numeric(rows, "selector_pred_pair_recall"),
        "selector_pred_pair_precision_mean": _summarize_numeric(rows, "selector_pred_pair_precision"),
        "selector_pred_pair_f1_mean": _summarize_numeric(rows, "selector_pred_pair_f1"),
        "selector_pred_site_f1_mean": _summarize_numeric(rows, "selector_pred_site_f1"),
        "actual_pair_recall_minus_pred_mean": _summarize_numeric(rows, "actual_pair_recall_minus_pred"),
        "actual_pair_precision_minus_pred_mean": _summarize_numeric(rows, "actual_pair_precision_minus_pred"),
        "actual_pair_f1_minus_pred_mean": _summarize_numeric(rows, "actual_pair_f1_minus_pred"),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-eval", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--replace-eval", type=Path, default=DEFAULT_REPLACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, action="append", default=None)
    args = parser.parse_args()

    budgets = sorted(set(args.budget or DEFAULT_BUDGETS))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        "diagnostic": args.diagnostic_eval,
        "replace": args.replace_eval,
    }
    stage_summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for stage, path in stages.items():
        summary, rows = _analyze_stage(stage, path, budgets)
        stage_summaries[stage] = summary
        all_rows.extend(rows)

    diag = stage_summaries.get("diagnostic", {})
    repl = stage_summaries.get("replace", {})
    cross_stage = {
        "replace_minus_diagnostic_delta_e_ratio": _as_float((repl.get("cumulative") or {}).get("delta_e_ratio"))
        - _as_float((diag.get("cumulative") or {}).get("delta_e_ratio")),
        "replace_minus_diagnostic_expected_time_ratio": _as_float((repl.get("cumulative") or {}).get("expected_time_ratio"))
        - _as_float((diag.get("cumulative") or {}).get("expected_time_ratio")),
        "replace_minus_diagnostic_tau_scale": _as_float((repl.get("tau_expected") or {}).get("scale_ratio"))
        - _as_float((diag.get("tau_expected") or {}).get("scale_ratio")),
        "replace_minus_diagnostic_site_f1": _as_float(repl.get("actual_site_f1_mean"))
        - _as_float(diag.get("actual_site_f1_mean")),
        "replace_minus_diagnostic_pair_f1": _as_float(repl.get("actual_pair_f1_mean"))
        - _as_float(diag.get("actual_pair_f1_mean")),
        "replace_minus_diagnostic_selected_pair_count": _as_float(repl.get("selected_pair_count_mean"))
        - _as_float(diag.get("selected_pair_count_mean")),
    }
    summary = {
        "stage": "v136_pair_retention_readonly",
        "mode": "pure-python read-only observed selected-overlap pair retention diagnostic",
        "limitations": [
            "v135 compact eval JSON exposes selected overlap pairs, not the full teacher vacancy-pair target per candidate.",
            "rank and retention metrics therefore describe observed selected true-pair overlap retention after budgeted projection.",
            "full true terminal pair rank still requires a non-compact rank diagnostic or teacher pair target export.",
        ],
        "budgets": budgets,
        "stages": stage_summaries,
        "cross_stage": cross_stage,
        "row_count": int(len(all_rows)),
        "artifacts": {
            "rows_jsonl": str(args.output_dir / "pair_retention_rows_v136.jsonl"),
            "summary_json": str(args.output_dir / "v136_pair_retention_readonly.json"),
            "stage_summary_json": str(args.output_dir / "stage_summary.json"),
        },
        "next_recommendation": (
            "If observed overlap pairs are already mostly retained but actual pair/site/energy remain poor, "
            "the failure is not the cap alone; it points back to pair-score calibration/state distribution shift. "
            "If observed overlap retention falls sharply below 128, a pair-score retention target is needed before another closed-loop selector."
        ),
    }

    row_path = args.output_dir / "pair_retention_rows_v136.jsonl"
    with row_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path = args.output_dir / "v136_pair_retention_readonly.json"
    stage_path = args.output_dir / "stage_summary.json"
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    summary_path.write_text(payload + "\n", encoding="utf-8")
    stage_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
