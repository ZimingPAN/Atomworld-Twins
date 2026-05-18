#!/usr/bin/env python3
"""Read-only v128 diagnostic for Pareto budget retention.

This script compares v123 predicted candidate budgets against the same-candidate
PR curve from v119 and the oracle candidate choices from v122. It is intentionally
pure Python so it can run on the remote host without touching torch.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_ROOT = Path("results/natural_teacher_support32_sequence_rollout_0517")
DEFAULT_V119 = DEFAULT_ROOT / "v119_pair_joint_selector_grouped_readonly_smoke" / "candidate_joint_targets_v119.jsonl"
DEFAULT_V122 = DEFAULT_ROOT / "v122_candidate_pareto_selector_grouped_readonly" / "candidate_pareto_choices_v122.jsonl"
DEFAULT_V123 = DEFAULT_ROOT / "v123_candidate_pareto_replay_readonly" / "candidate_pareto_replay_choices_v123.jsonl"
DEFAULT_OUTPUT = DEFAULT_ROOT / "v128_budget_retention_readonly"
DEFAULT_V127_STAGES = [
    DEFAULT_ROOT / "v127_candidate_pareto_budget_diagnostic_smoke20" / "stage_summary.json",
    DEFAULT_ROOT / "v127_candidate_pareto_budget_replace_smoke20" / "stage_summary.json",
    DEFAULT_ROOT / "v127_candidate_pareto_budget_add_smoke20" / "stage_summary.json",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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


def _candidate_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        _as_int(row.get("segment_index", row.get("group", 0))),
        _as_int(row.get("candidate_index", 0)),
        str(row.get("source_name", "")),
    )


def _budget_metrics(curve: dict[str, Any], budget: int) -> dict[str, float]:
    if not curve:
        return {}
    key = str(int(budget))
    if key not in curve:
        budgets = sorted((_as_int(item) for item in curve), key=lambda item: abs(item - int(budget)))
        key = str(budgets[0]) if budgets else key
    payload = curve.get(key)
    return payload if isinstance(payload, dict) else {}


def _min_budget_for_pair_recall(curve: dict[str, Any], floor: float) -> int | None:
    hits: list[int] = []
    for key, metrics in curve.items():
        if not isinstance(metrics, dict):
            continue
        if _as_float(metrics.get("pair_recall", 0.0)) >= float(floor):
            hits.append(_as_int(key))
    return min(hits) if hits else None


def _best_budget(row: dict[str, Any], target: str) -> int:
    best = row.get("best_budget", {})
    if isinstance(best, dict):
        return _as_int(best.get(target, 0))
    return 0


def _summarize_numbers(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _load_v127_summaries(paths: list[Path]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        eval_summary = data.get("eval", {})
        stage = str(data.get("stage", path.parent.name))
        summaries[path.parent.name] = {
            "stage": stage,
            "completed_segments": eval_summary.get("completed_segments"),
            "stop_reason": eval_summary.get("stop_reason"),
            "chosen_k_histogram": eval_summary.get("chosen_k_histogram"),
            "delta_e_ratio": (eval_summary.get("cumulative") or {}).get("delta_e_ratio"),
            "expected_time_ratio": (eval_summary.get("cumulative") or {}).get("expected_time_ratio"),
            "tau_scale": (eval_summary.get("tau_expected") or {}).get("scale_ratio"),
            "selected_overlap": data.get("selected_overlap", {}),
            "planner_candidate_pareto_selector": eval_summary.get("planner_candidate_pareto_selector", {}),
        }
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v119_candidate_jsonl", type=Path, default=DEFAULT_V119)
    parser.add_argument("--v122_choices_jsonl", type=Path, default=DEFAULT_V122)
    parser.add_argument("--v123_choices_jsonl", type=Path, default=DEFAULT_V123)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", action="append", default=None,
                        help="Predicted policy to inspect. Default inspects all v123 pred_* policies.")
    args = parser.parse_args()

    v119_rows = _load_jsonl(args.v119_candidate_jsonl)
    v122_choices = _load_jsonl(args.v122_choices_jsonl)
    v123_choices = _load_jsonl(args.v123_choices_jsonl)

    by_candidate = {_candidate_key(row): row for row in v119_rows}
    oracle_by_group = {
        _as_int(row.get("group")): row
        for row in v122_choices
        if row.get("policy") == "oracle_pareto_balanced"
    }

    policies = set(args.policy or [])
    if not policies:
        policies = {
            str(row.get("policy"))
            for row in v123_choices
            if str(row.get("policy", "")).startswith("pred_")
        }

    retention_rows: list[dict[str, Any]] = []
    missing_candidate_keys = 0
    for choice in v123_choices:
        policy = str(choice.get("policy", ""))
        if policy not in policies:
            continue
        group = _as_int(choice.get("group", 0))
        key = (
            group,
            _as_int(choice.get("candidate_index", 0)),
            str(choice.get("source_name", "")),
        )
        target_row = by_candidate.get(key)
        if target_row is None:
            missing_candidate_keys += 1
            continue
        curve = target_row.get("pr_curve", {})
        if not isinstance(curve, dict):
            curve = {}
        pred_budget = _as_int(choice.get("budget", choice.get("selected_pair_count", 0)))
        pred_metrics = _budget_metrics(curve, pred_budget)
        best_balanced_budget = _best_budget(target_row, "balanced_f1")
        best_pair_budget = _best_budget(target_row, "pair_f1")
        best_endpoint_budget = _best_budget(target_row, "endpoint_f1")
        best_balanced_metrics = _budget_metrics(curve, best_balanced_budget)
        best_pair_metrics = _budget_metrics(curve, best_pair_budget)
        oracle = oracle_by_group.get(group, {})
        oracle_same_candidate = (
            _as_int(oracle.get("candidate_index", -1)) == _as_int(choice.get("candidate_index", -2))
            and str(oracle.get("source_name", "")) == str(choice.get("source_name", ""))
        )
        row = {
            "group": group,
            "policy": policy,
            "candidate_index": _as_int(choice.get("candidate_index", 0)),
            "source_name": str(choice.get("source_name", "")),
            "segment_k": _as_int(choice.get("segment_k", 0)),
            "pred_budget": pred_budget,
            "same_candidate_best_balanced_budget": best_balanced_budget,
            "same_candidate_best_pair_budget": best_pair_budget,
            "same_candidate_best_endpoint_budget": best_endpoint_budget,
            "budget_minus_best_balanced": float(pred_budget - best_balanced_budget),
            "budget_ratio_to_best_balanced": float(pred_budget / max(best_balanced_budget, 1)),
            "pred_pair_precision": _as_float(pred_metrics.get("pair_precision", choice.get("pair_precision", 0.0))),
            "pred_pair_recall": _as_float(pred_metrics.get("pair_recall", choice.get("pair_recall", 0.0))),
            "pred_pair_f1": _as_float(pred_metrics.get("pair_f1", choice.get("pair_f1", 0.0))),
            "pred_endpoint_f1": _as_float(pred_metrics.get("endpoint_f1", choice.get("endpoint_f1", 0.0))),
            "same_candidate_best_balanced_pair_recall": _as_float(best_balanced_metrics.get("pair_recall", 0.0)),
            "same_candidate_best_balanced_pair_f1": _as_float(best_balanced_metrics.get("pair_f1", 0.0)),
            "same_candidate_best_balanced_endpoint_f1": _as_float(best_balanced_metrics.get("endpoint_f1", 0.0)),
            "same_candidate_best_pair_f1": _as_float(best_pair_metrics.get("pair_f1", 0.0)),
            "min_budget_pair_recall_0p5": _min_budget_for_pair_recall(curve, 0.5),
            "min_budget_pair_recall_0p8": _min_budget_for_pair_recall(curve, 0.8),
            "min_budget_pair_recall_1p0": _min_budget_for_pair_recall(curve, 1.0),
            "oracle_candidate_index": _as_int(oracle.get("candidate_index", -1)),
            "oracle_source_name": str(oracle.get("source_name", "")),
            "oracle_budget": _as_int(oracle.get("budget", 0)),
            "oracle_pair_precision": _as_float(oracle.get("pair_precision", 0.0)),
            "oracle_pair_recall": _as_float(oracle.get("pair_recall", 0.0)),
            "oracle_pair_f1": _as_float(oracle.get("pair_f1", 0.0)),
            "oracle_endpoint_f1": _as_float(oracle.get("endpoint_f1", 0.0)),
            "oracle_site_f1": _as_float(oracle.get("site_f1", 0.0)),
            "oracle_teacher_reward_sum": _as_float(oracle.get("teacher_reward_sum", 0.0)),
            "oracle_same_candidate": bool(oracle_same_candidate),
            "true_pair_count": _as_float(target_row.get("true_pair_count", 0.0)),
            "teacher_reward_sum": _as_float(choice.get("teacher_reward_sum", 0.0)),
            "site_f1": _as_float(choice.get("site_f1", 0.0)),
        }
        row["recall_gap_to_best_balanced"] = (
            row["same_candidate_best_balanced_pair_recall"] - row["pred_pair_recall"]
        )
        row["pair_f1_gap_to_best_balanced"] = (
            row["same_candidate_best_balanced_pair_f1"] - row["pred_pair_f1"]
        )
        for floor_key in ["0p5", "0p8", "1p0"]:
            min_budget = row[f"min_budget_pair_recall_{floor_key}"]
            row[f"pred_budget_below_recall_{floor_key}_budget"] = (
                bool(min_budget is not None and pred_budget < int(min_budget))
            )
        retention_rows.append(row)

    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retention_rows:
        by_policy[str(row["policy"])].append(row)

    policy_summary: dict[str, Any] = {}
    for policy, rows in sorted(by_policy.items()):
        count = len(rows)
        if count == 0:
            continue
        pred_recall = [float(row["pred_pair_recall"]) for row in rows]
        pred_pair_f1 = [float(row["pred_pair_f1"]) for row in rows]
        best_recall = [float(row["same_candidate_best_balanced_pair_recall"]) for row in rows]
        best_pair_f1 = [float(row["same_candidate_best_balanced_pair_f1"]) for row in rows]
        pred_budget = [float(row["pred_budget"]) for row in rows]
        best_budget = [float(row["same_candidate_best_balanced_budget"]) for row in rows]
        policy_summary[policy] = {
            "count": count,
            "pred_budget": _summarize_numbers(pred_budget),
            "same_candidate_best_balanced_budget": _summarize_numbers(best_budget),
            "budget_ratio_to_best_balanced": _summarize_numbers([
                float(row["budget_ratio_to_best_balanced"]) for row in rows
            ]),
            "pred_pair_precision": _summarize_numbers([float(row["pred_pair_precision"]) for row in rows]),
            "pred_pair_recall": _summarize_numbers(pred_recall),
            "pred_pair_f1": _summarize_numbers(pred_pair_f1),
            "same_candidate_best_balanced_pair_recall": _summarize_numbers(best_recall),
            "same_candidate_best_balanced_pair_f1": _summarize_numbers(best_pair_f1),
            "recall_gap_to_best_balanced_mean": float(mean([
                float(row["recall_gap_to_best_balanced"]) for row in rows
            ])),
            "pair_f1_gap_to_best_balanced_mean": float(mean([
                float(row["pair_f1_gap_to_best_balanced"]) for row in rows
            ])),
            "under_same_candidate_best_balanced_count": sum(
                int(float(row["pred_budget"]) < float(row["same_candidate_best_balanced_budget"]))
                for row in rows
            ),
            "pred_pair_recall_zero_count": sum(int(float(row["pred_pair_recall"]) <= 0.0) for row in rows),
            "pred_pair_recall_lt_0p5_count": sum(int(float(row["pred_pair_recall"]) < 0.5) for row in rows),
            "pred_pair_recall_lt_0p8_count": sum(int(float(row["pred_pair_recall"]) < 0.8) for row in rows),
            "pred_budget_below_min_recall_0p5_count": sum(
                int(row["pred_budget_below_recall_0p5_budget"]) for row in rows
            ),
            "pred_budget_below_min_recall_0p8_count": sum(
                int(row["pred_budget_below_recall_0p8_budget"]) for row in rows
            ),
            "pred_budget_below_min_recall_1p0_count": sum(
                int(row["pred_budget_below_recall_1p0_budget"]) for row in rows
            ),
            "oracle_same_candidate_count": sum(int(bool(row["oracle_same_candidate"])) for row in rows),
            "pred_budget_histogram": dict(sorted(Counter(_as_int(row["pred_budget"]) for row in rows).items())),
            "same_candidate_best_balanced_budget_histogram": dict(
                sorted(Counter(_as_int(row["same_candidate_best_balanced_budget"]) for row in rows).items())
            ),
        }

    focus = "pred_pareto_balanced"
    focus_rows = by_policy.get(focus, [])
    worst_focus_rows = sorted(
        focus_rows,
        key=lambda row: (
            float(row["recall_gap_to_best_balanced"]),
            -float(row["pred_pair_recall"]),
        ),
        reverse=True,
    )[:10]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / "budget_retention_rows_v128.jsonl"
    with row_path.open("w", encoding="utf-8") as handle:
        for row in retention_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "mode": "v128 pure-python predicted-budget vs oracle/true-pair-retention diagnostic",
        "input_files": {
            "v119_candidate_jsonl": str(args.v119_candidate_jsonl),
            "v122_choices_jsonl": str(args.v122_choices_jsonl),
            "v123_choices_jsonl": str(args.v123_choices_jsonl),
        },
        "output_files": {
            "budget_retention_rows": str(row_path),
            "summary": str(output_dir / "stage_summary.json"),
        },
        "row_count": int(len(retention_rows)),
        "missing_candidate_key_count": int(missing_candidate_keys),
        "policy_summary": policy_summary,
        "focus_policy": focus,
        "focus_policy_worst_recall_gap_preview": worst_focus_rows,
        "v127_closed_loop_summaries": _load_v127_summaries(DEFAULT_V127_STAGES),
        "current_judgement": (
            "If pred_pareto_balanced often chooses budgets below the same-candidate "
            "minimum recall floor or far below oracle best budgets, the v127 failure "
            "is a pruning target/calibration problem rather than a projection plumbing problem."
        ),
    }
    (output_dir / "stage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
