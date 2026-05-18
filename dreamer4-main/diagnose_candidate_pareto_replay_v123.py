#!/usr/bin/env python3
"""Export and replay the v122 Pareto candidate selector.

v122 found the first non-oracle loader-level policy that improved teacher
reward, terminal site overlap, pair precision/F1, and endpoint F1 together.
This script is the next default-off preflight step: it freezes the same
planner-visible feature construction and ridge coefficients into a selector
spec, then replays the policy over the existing 20 closed-loop candidate
groups.  It is intentionally pure Python and read-only; it does not train a
checkpoint or alter long-eval planner behavior.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import diagnose_candidate_pareto_selector_v122 as v122
import diagnose_candidate_feature_gap_v121 as v121


POLICIES_TO_REPLAY = (
    "selected_fixed_top32",
    "pred_pareto_balanced",
    "pred_soft_penalty_balanced",
    "pred_reward_floor_pair_first",
    "pred_reward_site_floor_pair_first",
    "pred_pair_floor_energy_site",
    "pred_two_branch_floor_balanced",
    "pred_strict_triple_floor",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    return v122._as_float(value, default)


def _mean(values: list[float]) -> float:
    return v122._mean(values)


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
        "version": "v123_candidate_pareto_replay",
        "description": (
            "Frozen v122 pred_pareto_balanced selector spec. The spec is "
            "for read-only replay/preflight; it is not yet a closed-loop "
            "long-eval planner integration."
        ),
        "feature_source": "combined_v120_plus_v104",
        "feature_dim": int(feature_dim),
        "train_row_count": int(train_row_count),
        "ridge_l2": float(ridge_l2),
        "predicted_targets": list(v122.PREDICTED_TARGETS),
        "v104_feature_order": list(v121.V104_FEATURE_ORDER),
        "base_feature_note": (
            "The first 34 features are inherited from v120/v118 PR-curve "
            "candidate-budget features; the final 11 are V104_FEATURE_ORDER."
        ),
        "policy": {
            "name": "pred_pareto_balanced",
            "pareto_keys": ["reward", "site", "pair_precision", "pair_f1", "endpoint"],
            "balanced_score_weights": {
                "reward": 0.24,
                "site": 0.22,
                "pair_f1": 0.22,
                "pair_precision": 0.18,
                "endpoint": 0.14,
            },
            "reward_prediction": "0.5 * teacher_reward_sum + 0.5 * reward_norm",
            "normalization": "per-candidate-group minmax over predictions",
        },
        "models": {name: _model_for_json(model) for name, model in sorted(models.items())},
        "join_summary": join_summary,
    }


def _group_rows(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["group"])].append(row)
    return grouped


def _summarize_policy_rows(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = v122._summarize_choices(rows)
    baseline = v122._summarize_choices(baseline_rows)
    folds_improving_all = 0
    for row, base in zip(rows, baseline_rows):
        if (
            v122._metric(row, "teacher_reward_sum") >= v122._metric(base, "teacher_reward_sum") - 1e-12
            and v122._metric(row, "site_f1") > v122._metric(base, "site_f1") + 1e-12
            and v122._metric(row, "pair_precision") > v122._metric(base, "pair_precision") + 1e-12
            and v122._metric(row, "pair_f1") > v122._metric(base, "pair_f1") + 1e-12
        ):
            folds_improving_all += 1
    return {
        "summary": summary,
        "minus_selected_fixed_top32": {
            "avg_teacher_reward_sum": _as_float(summary.get("avg_teacher_reward_sum", 0.0))
            - _as_float(baseline.get("avg_teacher_reward_sum", 0.0)),
            "avg_site_f1": _as_float(summary.get("avg_site_f1", 0.0)) - _as_float(baseline.get("avg_site_f1", 0.0)),
            "avg_pair_precision": _as_float(summary.get("avg_pair_precision", 0.0))
            - _as_float(baseline.get("avg_pair_precision", 0.0)),
            "avg_pair_f1": _as_float(summary.get("avg_pair_f1", 0.0)) - _as_float(baseline.get("avg_pair_f1", 0.0)),
            "avg_endpoint_f1": _as_float(summary.get("avg_endpoint_f1", 0.0))
            - _as_float(baseline.get("avg_endpoint_f1", 0.0)),
        },
        "folds_improving_reward_site_pair_precision_f1": int(folds_improving_all),
        "folds_total": int(len(rows)),
        "source_histogram": v122._source_hist(rows),
    }


def _row_preview(row: dict[str, Any], group: int, policy: str) -> dict[str, Any]:
    preview = v122._row_preview(row, group, policy)
    preview["global_replay"] = True
    preview["reward_norm"] = v122._metric(row, "reward_norm")
    preview["pair_recall"] = v122._metric(row, "pair_recall")
    return preview


def _replay_global_selector(rows: list[dict[str, Any]], models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    replay_rows = [copy.deepcopy(row) for row in rows]
    v122._attach_predictions(replay_rows, models)
    grouped = _group_rows(replay_rows)
    picks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    previews: list[dict[str, Any]] = []

    for group, group_rows in sorted(grouped.items()):
        policies = v122._pick_policies(group_rows)
        for policy in POLICIES_TO_REPLAY:
            if policy not in policies:
                continue
            row = policies[policy]
            picks[policy].append(row)
            previews.append(_row_preview(row, group, policy))

    baseline_rows = picks["selected_fixed_top32"]
    policy_eval = {
        policy: _summarize_policy_rows(policy_rows, baseline_rows)
        for policy, policy_rows in sorted(picks.items())
    }
    return {
        "policy_eval": policy_eval,
        "choice_rows": previews,
    }


def _load_v122_reference(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for policy in ("selected_fixed_top32", "pred_pareto_balanced", "pred_soft_penalty_balanced"):
        item = data.get("policy_eval", {}).get(policy, {})
        out[policy] = {
            "summary": item.get("summary", {}),
            "minus_selected_fixed_top32": item.get("minus_selected_fixed_top32", {}),
            "folds_improving_reward_site_pair_precision_f1": item.get(
                "folds_improving_reward_site_pair_precision_f1", 0
            ),
        }
    return out


def run(
    *,
    v119_candidate_jsonl: Path,
    v115_candidate_jsonl: Path,
    v104_candidate_jsonl: Path,
    output_dir: Path,
    ridge_l2: float,
    v122_reference_summary: Path | None,
) -> dict[str, Any]:
    rows, join_summary = v122._feature_rows(
        v119_candidate_jsonl=v119_candidate_jsonl,
        v115_candidate_jsonl=v115_candidate_jsonl,
        v104_candidate_jsonl=v104_candidate_jsonl,
    )
    if not rows:
        raise RuntimeError("no v123 selector rows")
    feature_dim = len(rows[0].get("features", []))
    models = v122._fit_models(rows, ridge_l2=ridge_l2)
    replay = _replay_global_selector(rows, models)

    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "candidate_pareto_selector_spec_v123.json"
    choices_path = output_dir / "candidate_pareto_replay_choices_v123.jsonl"
    spec = _selector_spec(
        models=models,
        feature_dim=feature_dim,
        ridge_l2=ridge_l2,
        train_row_count=len(rows),
        join_summary=join_summary,
    )
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    v122._write_jsonl(choices_path, replay["choice_rows"])

    policy_eval = replay["policy_eval"]
    baseline = policy_eval["selected_fixed_top32"]["summary"]
    best_non_oracle = max(
        (
            {"name": name, **item["summary"]}
            for name, item in policy_eval.items()
            if name != "selected_fixed_top32"
        ),
        key=lambda item: (
            _as_float(item.get("avg_teacher_reward_sum", 0.0))
            + _as_float(item.get("avg_site_f1", 0.0))
            + _as_float(item.get("avg_pair_precision", 0.0))
            + _as_float(item.get("avg_pair_f1", 0.0))
        ),
    )
    sync = [
        {"name": name, **item["summary"]}
        for name, item in sorted(policy_eval.items())
        if name != "selected_fixed_top32"
        and _as_float(item["summary"].get("avg_teacher_reward_sum", 0.0))
        >= _as_float(baseline.get("avg_teacher_reward_sum", 0.0)) - 1e-12
        and _as_float(item["summary"].get("avg_site_f1", 0.0))
        > _as_float(baseline.get("avg_site_f1", 0.0)) + 1e-12
        and _as_float(item["summary"].get("avg_pair_precision", 0.0))
        > _as_float(baseline.get("avg_pair_precision", 0.0)) + 1e-12
        and _as_float(item["summary"].get("avg_pair_f1", 0.0))
        > _as_float(baseline.get("avg_pair_f1", 0.0)) + 1e-12
    ]
    summary = {
        "mode": "v123 pure-python v122 selector export and read-only replay",
        "warning": (
            "Global replay trains the selector on all available candidate-budget rows, "
            "so it is a deployability/spec sanity check, not held-out evidence."
        ),
        "input_files": {
            "v119_candidate_jsonl": str(v119_candidate_jsonl),
            "v115_candidate_jsonl": str(v115_candidate_jsonl),
            "v104_candidate_jsonl": str(v104_candidate_jsonl),
            "v122_reference_summary": str(v122_reference_summary) if v122_reference_summary else "",
        },
        "candidate_budget_row_count": int(len(rows)),
        "candidate_count": int(len({tuple(row.get("key", ())) for row in rows})),
        "group_count": int(len(_group_rows(rows))),
        "feature_dim": int(feature_dim),
        "ridge_l2": float(ridge_l2),
        "join_summary": join_summary,
        "selector_spec_summary": {
            "path": str(spec_path),
            "model_targets": list(v122.PREDICTED_TARGETS),
            "policy": "pred_pareto_balanced",
        },
        "global_in_sample_replay": {
            "selected_fixed_top32": baseline,
            "best_non_oracle": best_non_oracle,
            "non_oracle_reward_site_pair_precision_f1_improvements": sync,
            "policy_eval": policy_eval,
        },
        "v122_heldout_reference": _load_v122_reference(v122_reference_summary),
        "current_judgement": (
            "Proceed to a true default-off long-eval candidate selector only if this "
            "spec can be computed from live planner candidates without teacher-label "
            "features. A closed-loop 20-segment smoke is still required before any "
            "80-segment confirmation or wider horizon."
        ),
        "output_files": {
            "summary": str(output_dir / "stage_summary.json"),
            "selector_spec": str(spec_path),
            "replay_choices": str(choices_path),
        },
    }
    (output_dir / "stage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v119-candidate-jsonl", type=Path, required=True)
    parser.add_argument("--v115-candidate-jsonl", type=Path, required=True)
    parser.add_argument("--v104-candidate-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--v122-reference-summary", type=Path, default=None)
    args = parser.parse_args()
    result = run(
        v119_candidate_jsonl=args.v119_candidate_jsonl,
        v115_candidate_jsonl=args.v115_candidate_jsonl,
        v104_candidate_jsonl=args.v104_candidate_jsonl,
        output_dir=args.output_dir,
        ridge_l2=float(args.ridge_l2),
        v122_reference_summary=args.v122_reference_summary,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
