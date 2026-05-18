#!/usr/bin/env python3
"""Pure-Python v121 planner-visible candidate feature gap diagnostic.

v120 showed that directly combining candidate-level energy/site quality with
pair-list pruning picks site/reward-looking candidates while sacrificing
terminal vacancy-pair precision. This read-only diagnostic asks a narrower
question before adding another trainable head: does adding planner-visible
candidate features from the v102-v104 diagnostics fix the selector, or is the
remaining blocker the target/modeling decomposition itself?

The script is intentionally torch-free. It does not modify checkpoints, caches,
or planner behavior.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import diagnose_candidate_pruning_selector_v120 as v120
import diagnose_vacancy_pair_pruning_v118 as v118


V104_FEATURE_ORDER = (
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


def _as_float(value: Any, default: float = 0.0) -> float:
    return v120._as_float(value, default)


def _mean(values: list[float]) -> float:
    return v120._mean(values)


def _candidate_key_no_source(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(v120._as_int(row.get("segment_index", 0))),
        int(v120._as_int(row.get("candidate_index", 0))),
        int(v120._as_int(row.get("segment_k", 0))),
    )


def _load_v104_features(path: Path) -> tuple[dict[tuple[int, int, int], list[float]], dict[str, Any]]:
    rows = v120._load_jsonl(path)
    features: dict[tuple[int, int, int], list[float]] = {}
    missing_by_name = {name: 0 for name in V104_FEATURE_ORDER}
    for row in rows:
        raw = row.get("features", {})
        if not isinstance(raw, dict):
            raw = {}
        values: list[float] = []
        for name in V104_FEATURE_ORDER:
            if name not in raw:
                missing_by_name[name] += 1
            values.append(_as_float(raw.get(name, 0.0)))
        features[_candidate_key_no_source(row)] = values
    return features, {
        "v104_candidate_count": int(len(rows)),
        "feature_order": list(V104_FEATURE_ORDER),
        "missing_by_feature": missing_by_name,
    }


def _budget_only_features(row: dict[str, Any]) -> list[float]:
    base = row.get("features", [])
    rows_count = int(_as_float(base[1], 0.0)) if isinstance(base, list) and len(base) > 1 else 0
    return v118._metric_features([], int(row.get("budget", 0)), rows_count)


def _with_feature_set(
    rows: list[dict[str, Any]],
    *,
    feature_name: str,
    v104_features: dict[tuple[int, int, int], list[float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    matched = 0
    for row in rows:
        item = copy.deepcopy(row)
        extra = v104_features.get(_candidate_key_no_source(row))
        if extra is not None:
            matched += 1
        else:
            extra = [0.0 for _ in V104_FEATURE_ORDER]

        base = list(row.get("features", []))
        budget_only = _budget_only_features(row)
        if feature_name == "base_v120_pair_pruning":
            item["features"] = base
        elif feature_name == "planner_visible_v104_only":
            item["features"] = list(extra) + budget_only
        elif feature_name == "combined_v120_plus_v104":
            item["features"] = base + list(extra)
        else:
            raise ValueError(f"unknown feature set: {feature_name}")
        out.append(item)

    return out, {
        "feature_set": feature_name,
        "candidate_budget_row_count": int(len(rows)),
        "matched_v104_feature_rows": int(matched),
        "join_coverage": float(matched / max(len(rows), 1)),
        "feature_dim": int(len(out[0]["features"]) if out else 0),
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-24 or vy <= 1e-24:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(vx * vy))


def _candidate_level_feature_correlations(
    expanded_rows: list[dict[str, Any]],
    v104_features: dict[tuple[int, int, int], list[float]],
) -> dict[str, Any]:
    # Collapse over budgets: the v104 features describe candidate/horizon
    # identity, not support-count choice.
    seen: set[tuple[str, int, int]] = set()
    rows: list[dict[str, Any]] = []
    for row in expanded_rows:
        key = tuple(row.get("key", ()))
        if key in seen:
            continue
        seen.add(key)
        extra = v104_features.get(_candidate_key_no_source(row))
        if extra is None:
            continue
        best_pair_f1 = 0.0
        best_endpoint_f1 = 0.0
        best_joint = 0.0
        for candidate_row in expanded_rows:
            if tuple(candidate_row.get("key", ())) != key:
                continue
            metrics = candidate_row["metrics"]
            targets = candidate_row["targets"]
            best_pair_f1 = max(best_pair_f1, _as_float(metrics.get("pair_f1", 0.0)))
            best_endpoint_f1 = max(best_endpoint_f1, _as_float(metrics.get("endpoint_f1", 0.0)))
            best_joint = max(best_joint, _as_float(targets.get("v120_joint_balanced", 0.0)))
        rows.append(
            {
                "features": extra,
                "site_f1": _as_float(row["metrics"].get("site_f1", 0.0)),
                "teacher_reward_sum": _as_float(row["metrics"].get("teacher_reward_sum", 0.0)),
                "best_pair_f1": best_pair_f1,
                "best_endpoint_f1": best_endpoint_f1,
                "best_joint": best_joint,
            }
        )

    targets = ("site_f1", "teacher_reward_sum", "best_pair_f1", "best_endpoint_f1", "best_joint")
    correlations: dict[str, dict[str, float]] = {}
    for target in targets:
        target_values = [_as_float(row[target]) for row in rows]
        values = {}
        for idx, name in enumerate(V104_FEATURE_ORDER):
            values[name] = _pearson([_as_float(row["features"][idx]) for row in rows], target_values)
        correlations[target] = dict(sorted(values.items(), key=lambda item: abs(item[1]), reverse=True))
    return {
        "candidate_count": int(len(rows)),
        "correlations": correlations,
    }


def _summarize_feature_eval(rows: list[dict[str, Any]], *, ridge_l2: float) -> dict[str, Any]:
    grouped = v120._fold_eval(rows, ridge_l2=ridge_l2)
    policies = grouped["policies"]
    selected_fixed = policies["selected_fixed_top32"]["summary"]
    best = v120._best_non_oracle(policies)
    return {
        "fold_count": int(grouped["fold_count"]),
        "selected_fixed_top32": selected_fixed,
        "best_non_oracle": best,
        "best_non_oracle_minus_selected_fixed_top32": {
            "avg_v120_joint_balanced": _as_float(best.get("avg_v120_joint_balanced", 0.0))
            - _as_float(selected_fixed.get("avg_v120_joint_balanced", 0.0)),
            "avg_site_f1": _as_float(best.get("avg_site_f1", 0.0))
            - _as_float(selected_fixed.get("avg_site_f1", 0.0)),
            "avg_teacher_reward_sum": _as_float(best.get("avg_teacher_reward_sum", 0.0))
            - _as_float(selected_fixed.get("avg_teacher_reward_sum", 0.0)),
            "avg_pair_f1": _as_float(best.get("avg_pair_f1", 0.0))
            - _as_float(selected_fixed.get("avg_pair_f1", 0.0)),
            "avg_pair_precision": _as_float(best.get("avg_pair_precision", 0.0))
            - _as_float(selected_fixed.get("avg_pair_precision", 0.0)),
            "avg_endpoint_f1": _as_float(best.get("avg_endpoint_f1", 0.0))
            - _as_float(selected_fixed.get("avg_endpoint_f1", 0.0)),
        },
        "non_oracle_policies_improving_site_and_pair_f1": v120._simultaneous_improvements(
            policies, selected_fixed
        ),
        "grouped_eval": grouped,
    }


def run(
    *,
    v119_candidate_jsonl: Path,
    v115_candidate_jsonl: Path,
    v104_candidate_jsonl: Path,
    output_dir: Path,
    ridge_l2: float,
) -> dict[str, Any]:
    v119_rows = v120._load_jsonl(v119_candidate_jsonl)
    v115_rows = v120._load_jsonl(v115_candidate_jsonl)
    joined = v120._join_candidate_rows(v119_rows, v115_rows)
    if not joined:
        raise RuntimeError("no joined v119/v115 candidate rows")
    v120._attach_group_targets(joined)
    expanded_rows = v120._candidate_budget_rows(joined)

    v104_features, v104_summary = _load_v104_features(v104_candidate_jsonl)
    feature_sets: dict[str, Any] = {}
    for feature_set in (
        "base_v120_pair_pruning",
        "planner_visible_v104_only",
        "combined_v120_plus_v104",
    ):
        rows_for_set, join_summary = _with_feature_set(
            expanded_rows,
            feature_name=feature_set,
            v104_features=v104_features,
        )
        feature_sets[feature_set] = {
            "join_summary": join_summary,
            "eval": _summarize_feature_eval(rows_for_set, ridge_l2=ridge_l2),
        }

    baseline = feature_sets["base_v120_pair_pruning"]["eval"]["selected_fixed_top32"]
    comparisons: dict[str, Any] = {}
    for name, item in feature_sets.items():
        best = item["eval"]["best_non_oracle"]
        comparisons[name] = {
            "best_non_oracle": best,
            "minus_selected_fixed_top32": item["eval"]["best_non_oracle_minus_selected_fixed_top32"],
            "has_non_oracle_site_pair_improvement": bool(
                item["eval"]["non_oracle_policies_improving_site_and_pair_f1"]
            ),
            "minus_base_best_non_oracle": {
                "avg_site_f1": _as_float(best.get("avg_site_f1", 0.0))
                - _as_float(feature_sets["base_v120_pair_pruning"]["eval"]["best_non_oracle"].get("avg_site_f1", 0.0)),
                "avg_pair_f1": _as_float(best.get("avg_pair_f1", 0.0))
                - _as_float(feature_sets["base_v120_pair_pruning"]["eval"]["best_non_oracle"].get("avg_pair_f1", 0.0)),
                "avg_pair_precision": _as_float(best.get("avg_pair_precision", 0.0))
                - _as_float(
                    feature_sets["base_v120_pair_pruning"]["eval"]["best_non_oracle"].get(
                        "avg_pair_precision", 0.0
                    )
                ),
                "avg_teacher_reward_sum": _as_float(best.get("avg_teacher_reward_sum", 0.0))
                - _as_float(
                    feature_sets["base_v120_pair_pruning"]["eval"]["best_non_oracle"].get(
                        "avg_teacher_reward_sum", 0.0
                    )
                ),
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "v121 pure-python planner-visible candidate feature gap diagnostic",
        "input_files": {
            "v119_candidate_jsonl": str(v119_candidate_jsonl),
            "v115_candidate_jsonl": str(v115_candidate_jsonl),
            "v104_candidate_jsonl": str(v104_candidate_jsonl),
        },
        "candidate_count": int(len(joined)),
        "candidate_budget_row_count": int(len(expanded_rows)),
        "group_count": int(len({int(row["group"]) for row in expanded_rows})),
        "ridge_l2": float(ridge_l2),
        "v104_feature_summary": v104_summary,
        "feature_correlations": _candidate_level_feature_correlations(expanded_rows, v104_features),
        "feature_sets": feature_sets,
        "feature_set_comparison": comparisons,
        "selected_fixed_top32_reference": baseline,
        "current_judgement": (
            "planner-visible v104 features close part of the feature gap only if a non-oracle policy "
            "improves site F1 and pair F1 over selected_fixed_top32 without collapsing reward. "
            "Otherwise the blocker remains target decomposition / pair interaction rather than "
            "missing simple planner-visible scalar features."
        ),
        "output_files": {
            "summary": str(output_dir / "stage_summary.json"),
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
    args = parser.parse_args()
    result = run(
        v119_candidate_jsonl=args.v119_candidate_jsonl,
        v115_candidate_jsonl=args.v115_candidate_jsonl,
        v104_candidate_jsonl=args.v104_candidate_jsonl,
        output_dir=args.output_dir,
        ridge_l2=float(args.ridge_l2),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
