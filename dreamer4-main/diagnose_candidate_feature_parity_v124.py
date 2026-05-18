#!/usr/bin/env python3
"""Read-only v124 feature-parity preflight for the v123 Pareto selector.

v123 froze a 45-dim selector spec from offline candidate-budget tables.  Before
that spec can be wired into long evaluation, this script checks whether every
feature block is reconstructable from live planner candidates without teacher
labels.  It is intentionally pure Python and read-only: no checkpoint loading,
no torch, no closed-loop rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CURVE_FEATURE_ORDER = (
    "bias",
    "pair_count",
    "segment_k",
    "candidate_index",
    "selected_by_planner",
    "source_is_vacancy",
    "source_is_energy",
    "source_is_factorized",
    "score_rank1",
    "score_rank2",
    "score_rank4",
    "score_rank8",
    "score_rank16",
    "score_rank32",
    "score_rank64",
    "score_rank128",
    "score_p50",
    "score_p75",
    "score_p90",
    "score_p95",
    "score_p99",
    "score_mean",
    "score_std",
    "score_gap_1_4",
    "score_gap_1_8",
    "score_gap_1_16",
    "score_gap_1_32",
    "score_gap_1_64",
    "score_gap_1_128",
)

BUDGET_FEATURE_ORDER = (
    "log_budget",
    "budget",
    "budget_fraction",
    "sqrt_budget",
    "inverse_budget",
)

DEFAULT_V104_FEATURE_ORDER = (
    "bias",
    "pre_oracle_selection_score_norm",
    "model_reward_sum_norm",
    "model_delta_e_norm",
    "model_tau_inv_norm",
    "model_noop_risk_inv_norm",
    "candidate_quality_score_norm",
    "projected_count_inv_norm",
    "vacancy_pair_count_inv_norm",
    "proposal_density_inv_norm",
    "segment_k_norm",
)

TEACHER_LABEL_FIELD_NAMES = (
    "best_budget",
    "candidate_joint_diagnostic",
    "endpoint_f1",
    "pair_f1",
    "pair_precision",
    "pair_recall",
    "pr_curve",
    "site_f1",
    "teacher_changed_count",
    "teacher_is_noop",
    "teacher_overlap_oracle",
    "teacher_pair_count",
    "teacher_reward_sum",
    "teacher_reward_per_sqrt_tau",
    "teacher_tau_exp",
    "true_pair_count",
    "vacancy_pair_f1",
    "vacancy_pair_precision",
    "vacancy_pair_recall",
)

LIVE_COMPACT_FIELDS = (
    "pre_oracle_selection_score",
    "predicted_reward_sum",
    "predicted_delta_e",
    "predicted_expected_tau",
    "predicted_noop_risk_prob",
    "candidate_quality_score",
    "projected_changed_count",
    "planner_edge_completion_support_count",
    "proposal_support_density",
    "segment_k",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _spec_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models", {})
    model_dims = {
        str(name): {
            "weights": len(_as_list(model.get("weights"))),
            "mean": len(_as_list(model.get("mean"))),
            "std": len(_as_list(model.get("std"))),
        }
        for name, model in sorted(models.items())
        if isinstance(model, dict)
    }
    v104_order = tuple(data.get("v104_feature_order") or DEFAULT_V104_FEATURE_ORDER)
    return {
        "path": str(path),
        "version": data.get("version", ""),
        "feature_source": data.get("feature_source", ""),
        "feature_dim": int(data.get("feature_dim", 0)),
        "predicted_targets": list(data.get("predicted_targets", [])),
        "v104_feature_order": list(v104_order),
        "model_dims": model_dims,
        "policy": data.get("policy", {}),
    }


def _v119_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_dims = sorted({len(row.get("features", [])) for row in rows if isinstance(row.get("features"), list)})
    top_level_keys = sorted({str(key) for row in rows for key in row.keys()})
    feature_values_are_vectors = all(isinstance(row.get("features"), list) for row in rows)
    label_keys_present = [key for key in TEACHER_LABEL_FIELD_NAMES if key in top_level_keys]
    return {
        "row_count": int(len(rows)),
        "feature_dim_unique": feature_dims,
        "feature_values_are_numeric_vectors": bool(feature_values_are_vectors),
        "top_level_label_fields_present_but_not_features": label_keys_present,
        "expected_curve_feature_dim": int(len(CURVE_FEATURE_ORDER)),
    }


def _v104_summary(rows: list[dict[str, Any]], expected_order: tuple[str, ...]) -> dict[str, Any]:
    feature_keys = sorted({str(key) for row in rows for key in (row.get("features") or {}).keys()})
    target_keys = sorted({str(key) for row in rows for key in (row.get("targets") or {}).keys()})
    missing = {
        name: sum(1 for row in rows if name not in (row.get("features") or {}))
        for name in expected_order
    }
    return {
        "row_count": int(len(rows)),
        "feature_keys": feature_keys,
        "target_keys_present_but_not_features": target_keys,
        "missing_by_feature": missing,
        "expected_v104_feature_dim": int(len(expected_order)),
    }


def _eval_source_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    probes = {
        "has_factorized_pair_scores_payload": "factorized_pair_scores" in text,
        "has_compact_candidate_output": "_planner_candidates_for_output" in text,
        "has_candidate_joint_teacher_diagnostic": "_candidate_joint_diagnostic_record" in text,
        "has_v123_selector_args": "candidate_pareto_selector_spec" in text or "planner_candidate_pareto" in text,
        "has_group_compact_fields": all(field in text for field in LIVE_COMPACT_FIELDS),
    }
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        **probes,
    }


def _feature_blocks(expected_v104_order: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": "pair_score_curve_features",
            "dim": len(CURVE_FEATURE_ORDER),
            "feature_order": list(CURVE_FEATURE_ORDER),
            "live_requirements": [
                "candidate segment_k and candidate_index",
                "legacy pre-selector selected_by_planner marker or an explicit false/default marker",
                "projection source family flags synthesized from planner source name",
                "full live pair score distribution, preferably vacancy_pair_projection_diagnostic.factorized_pair_scores",
            ],
            "teacher_label_required": False,
            "notes": (
                "This block used v118/v119 pair-score distribution features. "
                "It must be computed from live model pair scores, not from pr_curve, "
                "best_budget, true_pair_count, or teacher overlap fields."
            ),
        },
        {
            "name": "budget_features",
            "dim": len(BUDGET_FEATURE_ORDER),
            "feature_order": list(BUDGET_FEATURE_ORDER),
            "live_requirements": [
                "candidate pair_count",
                "candidate-budget sweep value",
            ],
            "teacher_label_required": False,
            "notes": "This block is deterministic once a live pair list and budget set exist.",
        },
        {
            "name": "planner_visible_v104_features",
            "dim": len(expected_v104_order),
            "feature_order": list(expected_v104_order),
            "live_requirements": list(LIVE_COMPACT_FIELDS) + ["per-candidate-group minmax normalization"],
            "teacher_label_required": False,
            "notes": (
                "These features are model/planner-visible if built from compact "
                "candidate fields before teacher probing. The normalization must be "
                "reimplemented over the current live candidate group."
            ),
        },
    ]


def run(
    *,
    selector_spec: Path,
    v119_candidate_jsonl: Path,
    v104_candidate_jsonl: Path,
    eval_source: Path,
    output_dir: Path,
) -> dict[str, Any]:
    spec = _spec_summary(selector_spec)
    expected_v104_order = tuple(spec.get("v104_feature_order") or DEFAULT_V104_FEATURE_ORDER)
    expected_dim = len(CURVE_FEATURE_ORDER) + len(BUDGET_FEATURE_ORDER) + len(expected_v104_order)
    v119_rows = _load_jsonl(v119_candidate_jsonl)
    v104_rows = _load_jsonl(v104_candidate_jsonl)
    v119 = _v119_summary(v119_rows)
    v104 = _v104_summary(v104_rows, expected_v104_order)
    eval_summary = _eval_source_summary(eval_source)
    model_dims_ok = all(
        item.get("mean") == expected_dim and item.get("std") == expected_dim and item.get("weights") == expected_dim + 1
        for item in spec.get("model_dims", {}).values()
    )
    feature_dim_ok = int(spec.get("feature_dim", 0)) == expected_dim
    artifact_feature_dims_ok = (
        v119.get("feature_dim_unique") == [len(CURVE_FEATURE_ORDER)]
        and all(value == 0 for value in v104.get("missing_by_feature", {}).values())
    )
    feature_blocks = _feature_blocks(expected_v104_order)
    forbidden_for_live_selector = list(TEACHER_LABEL_FIELD_NAMES)
    live_feature_parity = {
        "feature_dim_ok": bool(feature_dim_ok),
        "model_dims_ok": bool(model_dims_ok),
        "artifact_feature_dims_ok": bool(artifact_feature_dims_ok),
        "teacher_label_features_required": False,
        "requires_factorized_pair_score_payload": True,
        "requires_group_normalization": True,
        "requires_legacy_preselector_marker": True,
        "current_eval_has_raw_payloads": bool(
            eval_summary.get("has_factorized_pair_scores_payload")
            and eval_summary.get("has_compact_candidate_output")
            and eval_summary.get("has_group_compact_fields")
        ),
        "current_eval_has_selector_integration": bool(eval_summary.get("has_v123_selector_args")),
    }
    live_feature_parity["ready_for_closed_loop_selector_without_code_patch"] = bool(
        live_feature_parity["feature_dim_ok"]
        and live_feature_parity["model_dims_ok"]
        and live_feature_parity["artifact_feature_dims_ok"]
        and live_feature_parity["current_eval_has_raw_payloads"]
        and live_feature_parity["current_eval_has_selector_integration"]
    )
    if live_feature_parity["ready_for_closed_loop_selector_without_code_patch"]:
        judgement = (
            "The v123 45-dim feature vector appears live-computable and selector "
            "integration is already present; a guarded 20-segment smoke can be considered."
        )
    else:
        judgement = (
            "The v123 feature vector is label-clean in principle, but current long eval "
            "does not yet expose a default-off v123 selector integration. Add a guarded "
            "feature builder/selector path before any closed-loop smoke."
        )

    summary = {
        "mode": "v124 pure-python candidate selector feature-parity preflight",
        "selector_spec": spec,
        "expected_feature_layout": {
            "curve_dim": len(CURVE_FEATURE_ORDER),
            "budget_dim": len(BUDGET_FEATURE_ORDER),
            "v104_dim": len(expected_v104_order),
            "expected_total_dim": expected_dim,
        },
        "feature_blocks": feature_blocks,
        "source_artifact_checks": {
            "v119_candidate_jsonl": str(v119_candidate_jsonl),
            "v119_summary": v119,
            "v104_candidate_jsonl": str(v104_candidate_jsonl),
            "v104_summary": v104,
        },
        "eval_source_check": eval_summary,
        "forbidden_live_selector_fields": forbidden_for_live_selector,
        "live_feature_parity": live_feature_parity,
        "current_judgement": judgement,
        "next_step": (
            "Implement a default-off live candidate feature builder that computes "
            "the 34 pair/budget features from factorized pair scores and the 11 "
            "planner-visible features from compact candidate fields, then use the "
            "v123 spec for a 20-segment smoke only if the builder emits 45-dim rows "
            "without candidate_joint_diagnostic or teacher_overlap_oracle fields."
        ),
        "output_files": {
            "summary": str(output_dir / "stage_summary.json"),
            "feature_parity_report": str(output_dir / "candidate_feature_parity_report_v124.json"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "candidate_feature_parity_report_v124.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-spec", type=Path, required=True)
    parser.add_argument("--v119-candidate-jsonl", type=Path, required=True)
    parser.add_argument("--v104-candidate-jsonl", type=Path, required=True)
    parser.add_argument("--eval-source", type=Path, default=Path("eval_macro_long_trajectory.py"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        selector_spec=args.selector_spec,
        v119_candidate_jsonl=args.v119_candidate_jsonl,
        v104_candidate_jsonl=args.v104_candidate_jsonl,
        eval_source=args.eval_source,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
