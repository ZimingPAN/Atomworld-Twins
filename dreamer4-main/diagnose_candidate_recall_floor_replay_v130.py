#!/usr/bin/env python3
"""Export and replay the v129 recall-floor candidate-budget selector.

v129 showed that a predicted pair-recall floor is a safer support-count target
than the small v123 Pareto budget, but only at loader level.  This default-off
preflight freezes the same label-clean 45-D feature construction and global
ridge coefficients into a selector spec, then replays the policy over the
existing 20 candidate groups.  It is intentionally pure Python and read-only:
it does not train a checkpoint, start torch, or alter long-eval planner
behavior.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import diagnose_candidate_feature_gap_v121 as v121
import diagnose_candidate_recall_floor_v129 as v129
import diagnose_vacancy_pair_selector_v119 as v119


DEFAULT_ROOT = Path("results/natural_teacher_support32_sequence_rollout_0517")
DEFAULT_V119 = DEFAULT_ROOT / "v119_pair_joint_selector_grouped_readonly_smoke" / "candidate_joint_targets_v119.jsonl"
DEFAULT_V115 = DEFAULT_ROOT / "v115_pair_interaction_distill_readonly" / "candidate_support_count_samples_v115.jsonl"
DEFAULT_V104 = DEFAULT_ROOT / "v104_candidate_two_branch_selector_readonly_smoke20" / "candidate_two_branch_samples_v104.jsonl"
DEFAULT_V123 = DEFAULT_ROOT / "v123_candidate_pareto_replay_readonly" / "candidate_pareto_replay_choices_v123.jsonl"
DEFAULT_V129_SUMMARY = DEFAULT_ROOT / "v129_recall_floor_selector_readonly" / "stage_summary.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "v130_recall_floor_selector_replay_readonly"

POLICY_NAME = "pred_recall_floor0p6_balanced"
RECALL_FLOOR = 0.6


def _model_for_json(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "weights": [float(value) for value in model.get("weights", [])],
        "mean": [float(value) for value in model.get("mean", [])],
        "std": [float(value) for value in model.get("std", [])],
        "target_mean": float(model.get("target_mean", 0.0)),
        "target_std": float(model.get("target_std", 0.0)),
        "l2": float(model.get("l2", 0.0)),
    }


def _selector_spec(
    *,
    models: dict[str, dict[str, Any]],
    feature_dim: int,
    ridge_l2: float,
    train_row_count: int,
    join_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": "v130_recall_floor_selector_replay",
        "description": (
            "Frozen v129 predicted pair-recall-floor selector spec. The spec "
            "is for read-only replay/preflight; current long-eval Pareto "
            "integration must be extended before this policy is used in closed loop."
        ),
        "feature_source": "combined_v120_plus_v104",
        "feature_dim": int(feature_dim),
        "train_row_count": int(train_row_count),
        "ridge_l2": float(ridge_l2),
        "predicted_targets": list(v129.PREDICTED_TARGETS),
        "v104_feature_order": list(v121.V104_FEATURE_ORDER),
        "base_feature_note": (
            "The first 34 features are inherited from v120/v118 PR-curve "
            "candidate-budget features; the final 11 are V104_FEATURE_ORDER."
        ),
        "policy": {
            "name": POLICY_NAME,
            "pair_recall_floor": float(RECALL_FLOOR),
            "selection_rule": (
                "Within each candidate group, predict all target branches, keep "
                "rows with predicted pair_recall >= pair_recall_floor when any "
                "exist, then maximize the balanced score with lower selected "
                "pair count as the tie-breaker."
            ),
            "balanced_score_weights": {
                "reward": 0.24,
                "site": 0.22,
                "pair_precision": 0.18,
                "pair_f1": 0.22,
                "endpoint": 0.14,
            },
            "reward_prediction": "0.5 * teacher_reward_sum + 0.5 * reward_norm",
            "normalization": "per-candidate-group minmax over predictions",
            "fallback": "if no row satisfies the recall floor, score all rows",
        },
        "models": {name: _model_for_json(model) for name, model in sorted(models.items())},
        "join_summary": join_summary,
        "teacher_label_fields_used": False,
        "long_eval_policy_hook_ready": False,
        "long_eval_hook_note": (
            "eval_macro_long_trajectory.py can build the 45-D live features, "
            "but the current Pareto hook scores rows without enforcing this "
            "pair_recall floor. A default-off v131 hook is needed before closed-loop use."
        ),
    }


def _fit_global_models(rows: list[dict[str, Any]], *, ridge_l2: float) -> dict[str, dict[str, Any]]:
    xs = [row["features"] for row in rows]
    models: dict[str, dict[str, Any]] = {}
    for target in v129.PREDICTED_TARGETS:
        ys = [v129._target(row, target) for row in rows]
        models[target] = v119._fit_ridge_regressor(xs, ys, l2=ridge_l2)
    return models


def _attach_predictions(rows: list[dict[str, Any]], models: dict[str, dict[str, Any]]) -> None:
    xs = [row["features"] for row in rows]
    for target, model in models.items():
        preds = v119._predict_ridge(model, xs)
        for row, pred in zip(rows, preds):
            row.setdefault("predictions", {})[target] = float(pred)


def _group_rows(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[v129._as_int(row.get("group", 0))].append(row)
    return grouped


def _policy_preview(row: dict[str, Any], policy: str) -> dict[str, Any]:
    preview = v129._row_preview(row, policy)
    preview.update(
        {
            "pred_teacher_reward_sum": v129._pred(row, "teacher_reward_sum") if row.get("predictions") else None,
            "pred_site_f1": v129._pred(row, "site_f1") if row.get("predictions") else None,
            "pred_pair_precision": v129._pred(row, "pair_precision") if row.get("predictions") else None,
            "pred_pair_f1": v129._pred(row, "pair_f1") if row.get("predictions") else None,
            "pred_endpoint_f1": v129._pred(row, "endpoint_f1") if row.get("predictions") else None,
        }
    )
    return preview


def _same_candidate_v123_with_budget(rows: list[dict[str, Any]], choice: dict[str, Any], budget: int) -> dict[str, Any]:
    group = v129._as_int(choice.get("group", 0))
    candidate_index = v129._as_int(choice.get("candidate_index", 0))
    source_name = str(choice.get("source_name", ""))
    same = [
        row
        for row in rows
        if v129._as_int(row.get("group", 0)) == group
        and v129._as_int(row.get("candidate_index", 0)) == candidate_index
        and str(row.get("source_name", "")) == source_name
    ]
    if not same:
        return v129._selected_fixed(rows, budget=budget)
    return min(same, key=lambda row: abs(v129._as_int(row.get("budget", 0)) - int(budget)))


def _replay_global_selector(
    rows: list[dict[str, Any]],
    models: dict[str, dict[str, Any]],
    *,
    v123_choices_jsonl: Path,
) -> dict[str, Any]:
    replay_rows = copy.deepcopy(rows)
    _attach_predictions(replay_rows, models)
    grouped = _group_rows(replay_rows)
    v123_choices = [
        row for row in v129._load_jsonl(v123_choices_jsonl) if row.get("policy") == "pred_pareto_balanced"
    ]
    v123_by_group = {v129._as_int(row.get("group", 0)): row for row in v123_choices}

    picks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    previews: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        fixed = v129._selected_fixed(group_rows, budget=32)
        picks["selected_fixed_top32"].append(fixed)
        picks[POLICY_NAME].append(v129._pick_pred_recall_floor(group_rows, RECALL_FLOOR))
        picks["pred_recall_floor0p5_balanced"].append(v129._pick_pred_recall_floor(group_rows, 0.5))
        choice = v123_by_group.get(group)
        if choice is not None:
            picks["v123_pred_pareto_same_candidate_top32"].append(
                _same_candidate_v123_with_budget(group_rows, choice, budget=32)
            )
        for policy, policy_rows in sorted(picks.items()):
            if len(policy_rows) and policy_rows[-1] in group_rows:
                previews.append(_policy_preview(policy_rows[-1], policy))

    baseline = picks["selected_fixed_top32"]
    policy_eval = {policy: v129._summarize(policy_rows, baseline) for policy, policy_rows in sorted(picks.items())}
    return {
        "policy_eval": policy_eval,
        "choices": previews,
        "rows_with_predictions": replay_rows,
    }


def _prediction_quality(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        target: v129._corr([v129._pred(row, target) for row in rows], [v129._target(row, target) for row in rows])
        for target in v129.PREDICTED_TARGETS
    }


def run(
    *,
    v119_candidate_jsonl: Path,
    v115_candidate_jsonl: Path,
    v104_candidate_jsonl: Path,
    v123_choices_jsonl: Path,
    v129_summary_json: Path | None,
    output_dir: Path,
    ridge_l2: float,
) -> dict[str, Any]:
    rows, join_summary = v129._load_feature_rows(
        v119_candidate_jsonl=v119_candidate_jsonl,
        v115_candidate_jsonl=v115_candidate_jsonl,
        v104_candidate_jsonl=v104_candidate_jsonl,
    )
    feature_dims = Counter(len(row.get("features", [])) for row in rows)
    if set(feature_dims) != {45}:
        raise RuntimeError(f"v130 expects only 45-D features, got {dict(feature_dims)}")

    models = _fit_global_models(rows, ridge_l2=ridge_l2)
    replay = _replay_global_selector(rows, models, v123_choices_jsonl=v123_choices_jsonl)
    predicted_rows = replay["rows_with_predictions"]
    policy_eval = replay["policy_eval"]

    v129_summary = (
        json.loads(v129_summary_json.read_text(encoding="utf-8"))
        if v129_summary_json and v129_summary_json.exists()
        else {}
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "candidate_recall_floor_selector_spec_v130.json"
    choices_path = output_dir / "candidate_recall_floor_replay_choices_v130.jsonl"
    summary_path = output_dir / "stage_summary.json"

    spec = _selector_spec(
        models=models,
        feature_dim=45,
        ridge_l2=ridge_l2,
        train_row_count=len(rows),
        join_summary=join_summary,
    )
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    v129._write_jsonl(choices_path, replay["choices"])

    baseline = policy_eval.get("selected_fixed_top32", {})
    selected = policy_eval.get(POLICY_NAME, {})
    improvement = selected.get("minus_selected_fixed_top32", {})
    summary = {
        "mode": "v130 pure-python recall-floor selector spec export and replay",
        "input_files": {
            "v119_candidate_jsonl": str(v119_candidate_jsonl),
            "v115_candidate_jsonl": str(v115_candidate_jsonl),
            "v104_candidate_jsonl": str(v104_candidate_jsonl),
            "v123_choices_jsonl": str(v123_choices_jsonl),
            "v129_summary_json": str(v129_summary_json) if v129_summary_json else "",
        },
        "candidate_budget_row_count": len(rows),
        "candidate_count": len({tuple(row.get("key", ())) for row in rows}),
        "group_count": len(_group_rows(rows)),
        "feature_dim_histogram": {str(k): int(v) for k, v in sorted(feature_dims.items())},
        "ridge_l2": float(ridge_l2),
        "join_summary": join_summary,
        "spec_written": True,
        "teacher_label_fields_used": False,
        "long_eval_policy_hook_ready": False,
        "selected_fixed_top32": baseline,
        POLICY_NAME: selected,
        "policy_eval": policy_eval,
        "prediction_quality_corr_insample": _prediction_quality(predicted_rows),
        "v129_grouped_reference_best_non_oracle": v129_summary.get("best_non_oracle", {}),
        "v129_grouped_reference_prediction_quality_corr": v129_summary.get("prediction_quality_corr", {}),
        "deployment_preflight": {
            "feature_dim_ok": set(feature_dims) == {45},
            "spec_has_pair_recall_model": "pair_recall" in spec.get("models", {}),
            "old_selection_unchanged_by_this_script": True,
            "requires_default_off_long_eval_policy_hook": True,
            "budget_pruning_not_run_here": True,
        },
        "output_files": {
            "spec": str(spec_path),
            "choices": str(choices_path),
            "summary": str(summary_path),
        },
        "current_judgement": (
            "v130 only freezes and replays the v129 recall-floor selector. "
            "It is useful for feature/spec preflight, but it is not a closed-loop result. "
            "A separate default-off long-eval hook must enforce predicted pair_recall >= 0.6 "
            "before any 20-segment smoke."
        ),
    }
    if improvement:
        summary["average_improvement_over_fixed_top32"] = improvement

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v119-candidate-jsonl", type=Path, default=DEFAULT_V119)
    parser.add_argument("--v115-candidate-jsonl", type=Path, default=DEFAULT_V115)
    parser.add_argument("--v104-candidate-jsonl", type=Path, default=DEFAULT_V104)
    parser.add_argument("--v123-choices-jsonl", type=Path, default=DEFAULT_V123)
    parser.add_argument("--v129-summary-json", type=Path, default=DEFAULT_V129_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    args = parser.parse_args()
    summary = run(
        v119_candidate_jsonl=args.v119_candidate_jsonl,
        v115_candidate_jsonl=args.v115_candidate_jsonl,
        v104_candidate_jsonl=args.v104_candidate_jsonl,
        v123_choices_jsonl=args.v123_choices_jsonl,
        v129_summary_json=args.v129_summary_json,
        output_dir=args.output_dir,
        ridge_l2=args.ridge_l2,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
