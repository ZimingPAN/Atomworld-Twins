#!/usr/bin/env python3
"""Read-only v133 conservative calibration / budget-guard diagnostic.

v132 showed that the v130 recall-floor selector is badly miscalibrated on live
v131 candidates: tiny budgets pass the recall floor with predicted pair recall
well above one.  This script tests default-off guard policies on the same live
diagnostic rows before any closed-loop selector smoke:

* prediction clipping;
* conservative minimum budget floors;
* live pair-score feature z-score normalization;
* pair-recall rescaling using offline-vs-live diagnostic means.

It is intentionally pure Python and read-only.  It does not run torch, train a
checkpoint, or invoke the long evaluator.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import diagnose_candidate_live_drift_v132 as v132


DEFAULT_ROOT = Path("results/natural_teacher_support32_sequence_rollout_0517")
DEFAULT_V119 = DEFAULT_ROOT / "v119_pair_joint_selector_grouped_readonly_smoke" / "candidate_joint_targets_v119.jsonl"
DEFAULT_V115 = DEFAULT_ROOT / "v115_pair_interaction_distill_readonly" / "candidate_support_count_samples_v115.jsonl"
DEFAULT_V104 = DEFAULT_ROOT / "v104_candidate_two_branch_selector_readonly_smoke20" / "candidate_two_branch_samples_v104.jsonl"
DEFAULT_V130_SPEC = DEFAULT_ROOT / "v130_recall_floor_selector_replay_readonly" / "candidate_recall_floor_selector_spec_v130.json"
DEFAULT_V130_SUMMARY = DEFAULT_ROOT / "v130_recall_floor_selector_replay_readonly" / "stage_summary.json"
DEFAULT_V131_EVAL = DEFAULT_ROOT / "v131_recall_floor_selector_diagnostic_smoke20" / "eval_long_recall_floor_diagnostic_20.json"
DEFAULT_V131_SUMMARY = DEFAULT_ROOT / "v131_recall_floor_selector_diagnostic_smoke20" / "stage_summary.json"
DEFAULT_V132_SUMMARY = DEFAULT_ROOT / "v132_live_calibration_feature_drift_readonly" / "stage_summary.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "v133_conservative_budget_guard_readonly"

PROBABILITY_TARGETS = ("site_f1", "pair_precision", "pair_recall", "pair_f1", "endpoint_f1")
PAIR_SCORE_FEATURE_INDICES = tuple(range(8, 29))


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
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(mean(finite)) if finite else 0.0


def _std(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0
    mu = _mean(finite)
    return float(math.sqrt(sum((value - mu) ** 2 for value in finite) / len(finite)))


def _quantile(values: list[float], q: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return 0.0
    idx = min(max(int(round(float(q) * (len(finite) - 1))), 0), len(finite) - 1)
    return float(finite[idx])


def _hist(values: list[Any]) -> dict[str, int]:
    counter = Counter(str(value) for value in values)

    def key_fn(item: tuple[str, int]) -> tuple[int, float | str]:
        key = item[0]
        try:
            return (0, float(key))
        except ValueError:
            return (1, key)

    return dict(sorted(counter.items(), key=key_fn))


def _metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": _mean(values),
        "std": _std(values),
        "min": _quantile(values, 0.0),
        "p25": _quantile(values, 0.25),
        "p50": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "max": _quantile(values, 1.0),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _attach_predictions(rows: list[dict[str, Any]], spec: dict[str, Any]) -> None:
    for row in rows:
        row["predictions"] = {
            target: v132._predict(spec, target, [float(value) for value in row.get("features", [])])
            for target in v132.PARETO_TARGETS
        }


def _group_rows(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_as_int(row.get("group", 0))].append(row)
    return grouped


def _prediction_stats(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    return {
        target: v132._minmax([
            _as_float(row.get("predictions", {}).get(target, 0.0))
            for row in rows
        ])
        for target in v132.PARETO_TARGETS
    }


def _score(row: dict[str, Any], stats: dict[str, tuple[float, float]]) -> float:
    return v132._balanced_score(row, stats)


def _clip_predictions(row: dict[str, Any]) -> None:
    preds = row.setdefault("predictions", {})
    for target in PROBABILITY_TARGETS:
        preds[target] = min(max(_as_float(preds.get(target, 0.0)), 0.0), 1.0)


def _scale_pair_recall(row: dict[str, Any], scale: float, *, clip: bool = True) -> None:
    preds = row.setdefault("predictions", {})
    value = _as_float(preds.get("pair_recall", 0.0)) * float(scale)
    preds["pair_recall"] = min(max(value, 0.0), 1.0) if clip else value


def _normalize_pair_score_features(
    rows: list[dict[str, Any]],
    *,
    offline_rows: list[dict[str, Any]],
    spec: dict[str, Any],
) -> None:
    offline_stats: dict[int, tuple[float, float]] = {}
    live_stats: dict[int, tuple[float, float]] = {}
    for idx in PAIR_SCORE_FEATURE_INDICES:
        offline_values = [
            _as_float(row.get("features", [])[idx], 0.0)
            for row in offline_rows
            if len(row.get("features", [])) > idx
        ]
        live_values = [
            _as_float(row.get("features", [])[idx], 0.0)
            for row in rows
            if len(row.get("features", [])) > idx
        ]
        offline_stats[idx] = (_mean(offline_values), max(_std(offline_values), 1e-12))
        live_stats[idx] = (_mean(live_values), max(_std(live_values), 1e-12))
    for row in rows:
        features = [float(value) for value in row.get("features", [])]
        for idx in PAIR_SCORE_FEATURE_INDICES:
            if idx >= len(features):
                continue
            offline_mu, offline_std = offline_stats[idx]
            live_mu, live_std = live_stats[idx]
            features[idx] = offline_mu + (features[idx] - live_mu) * (offline_std / live_std)
        row["features"] = features
    _attach_predictions(rows, spec)


def _policy_pick(
    group_rows: list[dict[str, Any]],
    *,
    recall_floor: float,
    min_budget: int | None = None,
    prefer_lower_budget: bool = True,
) -> dict[str, Any]:
    stats = _prediction_stats(group_rows)
    for row in group_rows:
        row["guard_selector_score"] = _score(row, stats)
    pool = [
        row
        for row in group_rows
        if _as_float(row.get("predictions", {}).get("pair_recall", 0.0)) >= recall_floor
        and (min_budget is None or _as_int(row.get("budget", 0)) >= int(min_budget))
    ]
    if not pool and min_budget is not None:
        pool = [row for row in group_rows if _as_int(row.get("budget", 0)) >= int(min_budget)]
    if not pool:
        pool = list(group_rows)
    return max(
        pool,
        key=lambda row: (
            _as_float(row.get("guard_selector_score", 0.0)),
            -_as_float(row.get("budget", 0.0)) if prefer_lower_budget else _as_float(row.get("budget", 0.0)),
        ),
    )


def _apply_guard_policy(
    rows: list[dict[str, Any]],
    *,
    name: str,
    spec: dict[str, Any],
    offline_rows: list[dict[str, Any]],
    recall_floor: float,
    min_budget: int | None = None,
    clip_predictions: bool = False,
    pair_recall_scale: float | None = None,
    normalize_pair_scores: bool = False,
) -> list[dict[str, Any]]:
    policy_rows = copy.deepcopy(rows)
    if normalize_pair_scores:
        _normalize_pair_score_features(policy_rows, offline_rows=offline_rows, spec=spec)
    if clip_predictions or pair_recall_scale is not None:
        for row in policy_rows:
            if pair_recall_scale is not None:
                _scale_pair_recall(row, pair_recall_scale, clip=True)
            if clip_predictions:
                _clip_predictions(row)
    picks: list[dict[str, Any]] = []
    for _, group_rows in sorted(_group_rows(policy_rows).items()):
        pick = _policy_pick(group_rows, recall_floor=recall_floor, min_budget=min_budget)
        pick = copy.deepcopy(pick)
        pick["policy"] = name
        pick["min_budget"] = min_budget
        pick["clip_predictions"] = bool(clip_predictions)
        pick["pair_recall_scale"] = pair_recall_scale
        pick["normalize_pair_scores"] = bool(normalize_pair_scores)
        picks.append(pick)
    return picks


def _summarize_picks(picks: list[dict[str, Any]]) -> dict[str, Any]:
    pair_recall_values = [_as_float(row.get("predictions", {}).get("pair_recall", 0.0)) for row in picks]
    budgets = [_as_int(row.get("budget", 0)) for row in picks]
    pair_counts = [_as_float(row.get("selected_pair_count", row.get("budget", 0.0))) for row in picks]
    return {
        "count": len(picks),
        "budget_histogram": _hist(budgets),
        "avg_budget": _mean([float(value) for value in budgets]),
        "avg_selected_pair_count": _mean(pair_counts),
        "min_budget": min(budgets) if budgets else 0,
        "max_budget": max(budgets) if budgets else 0,
        "selected_k_histogram": _hist([_as_int(row.get("segment_k", 0)) for row in picks]),
        "selected_by_preselector_mean": _mean([1.0 if row.get("selected_by_preselector") else 0.0 for row in picks]),
        "pred_pair_recall": _metric_summary(pair_recall_values),
        "pred_pair_recall_gt_1_count": sum(1 for value in pair_recall_values if value > 1.0),
        "pred_pair_recall_lt_0_count": sum(1 for value in pair_recall_values if value < 0.0),
        "pred_pair_recall_floor_pass_count": sum(1 for value in pair_recall_values if value >= 0.6),
        "pred_site_f1": _metric_summary([_as_float(row.get("predictions", {}).get("site_f1", 0.0)) for row in picks]),
        "pred_pair_precision": _metric_summary([_as_float(row.get("predictions", {}).get("pair_precision", 0.0)) for row in picks]),
        "pred_pair_f1": _metric_summary([_as_float(row.get("predictions", {}).get("pair_f1", 0.0)) for row in picks]),
        "pred_endpoint_f1": _metric_summary([_as_float(row.get("predictions", {}).get("endpoint_f1", 0.0)) for row in picks]),
        "guard_pass_no_tiny_high_recall": (
            sum(1 for row in picks if _as_int(row.get("budget", 0)) <= 12 and _as_float(row.get("predictions", {}).get("pair_recall", 0.0)) >= 0.6)
            == 0
        ),
        "guard_pass_no_pair_recall_gt_1": all(value <= 1.0 for value in pair_recall_values),
    }


def run(
    *,
    v119_candidate_jsonl: Path,
    v115_candidate_jsonl: Path,
    v104_candidate_jsonl: Path,
    v130_spec_json: Path,
    v130_summary_json: Path,
    v131_eval_json: Path,
    v131_summary_json: Path,
    v132_summary_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    spec = v132._load_json(v130_spec_json)
    v130_summary = v132._load_json(v130_summary_json)
    v131_summary = v132._load_json(v131_summary_json)
    v132_summary = v132._load_json(v132_summary_json)
    eval_payload = v132._load_json(v131_eval_json)
    offline_rows, offline_policy_picks = v132._offline_rows(
        v119_candidate_jsonl=v119_candidate_jsonl,
        v115_candidate_jsonl=v115_candidate_jsonl,
        v104_candidate_jsonl=v104_candidate_jsonl,
        spec=spec,
    )
    live_rows, _, _, _ = v132._live_rows(eval_payload, spec)

    offline_policy_mean_recall = _mean([
        _as_float(row.get("predictions", {}).get("pair_recall", 0.0))
        for row in offline_policy_picks
    ])
    live_unguarded_picks = _apply_guard_policy(
        live_rows,
        name="live_unguarded_recall_floor0p6",
        spec=spec,
        offline_rows=offline_rows,
        recall_floor=0.6,
    )
    live_unguarded_mean_recall = _mean([
        _as_float(row.get("predictions", {}).get("pair_recall", 0.0))
        for row in live_unguarded_picks
    ])
    recall_scale_to_offline_policy = offline_policy_mean_recall / max(live_unguarded_mean_recall, 1e-12)

    policies: dict[str, list[dict[str, Any]]] = {}
    policy_configs = [
        {"name": "live_unguarded_recall_floor0p6"},
        {"name": "clip_probability_predictions", "clip_predictions": True},
        {
            "name": "pair_recall_rescaled_to_v130_policy_mean",
            "pair_recall_scale": recall_scale_to_offline_policy,
            "clip_predictions": True,
        },
        {"name": "min_budget_32", "min_budget": 32, "clip_predictions": True},
        {"name": "min_budget_48", "min_budget": 48, "clip_predictions": True},
        {"name": "min_budget_64", "min_budget": 64, "clip_predictions": True},
        {
            "name": "rescaled_pair_recall_min_budget_48",
            "pair_recall_scale": recall_scale_to_offline_policy,
            "min_budget": 48,
            "clip_predictions": True,
        },
        {
            "name": "score_scale_normalized",
            "normalize_pair_scores": True,
            "clip_predictions": True,
        },
        {
            "name": "score_scale_normalized_min_budget_48",
            "normalize_pair_scores": True,
            "min_budget": 48,
            "clip_predictions": True,
        },
    ]
    for config in policy_configs:
        policies[config["name"]] = _apply_guard_policy(
            live_rows,
            spec=spec,
            offline_rows=offline_rows,
            recall_floor=0.6,
            **config,
        )

    summaries = {name: _summarize_picks(picks) for name, picks in policies.items()}
    viable = [
        name
        for name, summary in summaries.items()
        if summary["guard_pass_no_tiny_high_recall"] and summary["guard_pass_no_pair_recall_gt_1"]
    ]
    conservative = [
        name
        for name, summary in summaries.items()
        if summary["guard_pass_no_tiny_high_recall"]
        and summary["guard_pass_no_pair_recall_gt_1"]
        and summary["avg_budget"] >= 32.0
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, picks in policies.items():
        _write_jsonl(output_dir / f"{name}_picks_v133.jsonl", picks)

    summary = {
        "mode": "v133 pure-python conservative calibration / budget-guard readonly diagnostic",
        "input_files": {
            "v119_candidate_jsonl": str(v119_candidate_jsonl),
            "v115_candidate_jsonl": str(v115_candidate_jsonl),
            "v104_candidate_jsonl": str(v104_candidate_jsonl),
            "v130_spec_json": str(v130_spec_json),
            "v130_summary_json": str(v130_summary_json),
            "v131_eval_json": str(v131_eval_json),
            "v131_summary_json": str(v131_summary_json),
            "v132_summary_json": str(v132_summary_json),
        },
        "doc_gap": {
            "doc_atomworld_mirror_md_missing": not Path("../doc/AtomWorld-Mirror.md").exists(),
        },
        "reference": {
            "v130_pred_recall_floor0p6_balanced": v130_summary.get("pred_recall_floor0p6_balanced", {}),
            "v131_selector_diagnostic": (v131_summary.get("eval", {}) or {}).get("planner_candidate_pareto_selector", {}),
            "v132_drift_core": {
                key: v132_summary.get("drift", {}).get(key)
                for key in [
                    "offline_policy_avg_budget",
                    "live_recorded_selector_avg_budget",
                    "offline_policy_avg_pred_pair_recall",
                    "live_recorded_selector_avg_pred_pair_recall",
                    "live_recorded_pair_recall_gt_1_count",
                ]
            },
            "recall_scale_to_offline_policy_mean": recall_scale_to_offline_policy,
        },
        "policy_summaries": summaries,
        "viable_non_tiny_no_gt1_policies": viable,
        "conservative_non_tiny_no_gt1_policies": conservative,
        "current_judgement": (
            "v133 is a read-only guard preflight. It does not prove closed-loop improvement. "
            "A policy is only useful as the next preflight if it prevents tiny-budget high-recall "
            "misclassification and avoids pair_recall predictions above one on live v131 candidates."
        ),
    }
    summary_path = output_dir / "stage_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v119_candidate_jsonl", type=Path, default=DEFAULT_V119)
    parser.add_argument("--v115_candidate_jsonl", type=Path, default=DEFAULT_V115)
    parser.add_argument("--v104_candidate_jsonl", type=Path, default=DEFAULT_V104)
    parser.add_argument("--v130_spec_json", type=Path, default=DEFAULT_V130_SPEC)
    parser.add_argument("--v130_summary_json", type=Path, default=DEFAULT_V130_SUMMARY)
    parser.add_argument("--v131_eval_json", type=Path, default=DEFAULT_V131_EVAL)
    parser.add_argument("--v131_summary_json", type=Path, default=DEFAULT_V131_SUMMARY)
    parser.add_argument("--v132_summary_json", type=Path, default=DEFAULT_V132_SUMMARY)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = run(
        v119_candidate_jsonl=args.v119_candidate_jsonl,
        v115_candidate_jsonl=args.v115_candidate_jsonl,
        v104_candidate_jsonl=args.v104_candidate_jsonl,
        v130_spec_json=args.v130_spec_json,
        v130_summary_json=args.v130_summary_json,
        v131_eval_json=args.v131_eval_json,
        v131_summary_json=args.v131_summary_json,
        v132_summary_json=args.v132_summary_json,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
