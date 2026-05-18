#!/usr/bin/env python3
"""Pure-Python v120 candidate-quality + pair-pruning selector diagnostic.

v119 showed that a learned pair score plus PR-curve pruning can beat fixed
top32, but it optimizes only pair-list support. This read-only diagnostic joins
the v119 pair-budget targets with v115 candidate-level teacher probes and tests
whether a grouped selector can choose both:

* the candidate/horizon with better teacher reward and terminal site support
* the pair-list budget with better terminal vacancy-pair precision/F1

It is intentionally torch-free and does not modify checkpoints or planner
behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import diagnose_vacancy_pair_pruning_v118 as v118
import diagnose_vacancy_pair_selector_v119 as v119


OBJECTIVES = (
    "v120_joint_balanced",
    "v120_energy_site_pair",
    "v120_precision_first",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    return v118._as_float(value, default)


def _as_int(value: Any, default: int = 0) -> int:
    return v118._as_int(value, default)


def _mean(values: list[float]) -> float:
    return v118._mean(values)


def _std(values: list[float]) -> float:
    return v118._std(values)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return v118._load_jsonl(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _candidate_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return v118._candidate_key(row)


def _group_key(row: dict[str, Any]) -> int:
    return v118._group_key(row)


def _minmax(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return lo, lo + 1.0
    return lo, hi


def _norm(value: float, lo: float, hi: float) -> float:
    return float((float(value) - lo) / max(float(hi) - float(lo), 1e-12))


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _join_candidate_rows(
    v119_rows: list[dict[str, Any]],
    candidate_probe_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    probe_by_key = {_candidate_key(row): row for row in candidate_probe_rows}
    joined: list[dict[str, Any]] = []
    missing = 0
    for row in v119_rows:
        probe = probe_by_key.get(_candidate_key(row))
        if probe is None:
            missing += 1
            continue
        item = dict(row)
        item["probe"] = probe
        joined.append(item)
    if missing:
        print(f"[v120] skipped {missing} v119 candidate rows without v115 probe rows")
    return joined


def _attach_group_targets(rows: list[dict[str, Any]]) -> None:
    by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[_group_key(row)].append(row)
    for group_rows in by_group.values():
        rewards = [_as_float(row["probe"].get("teacher_reward_sum", 0.0)) for row in group_rows]
        reward_lo, reward_hi = _minmax(rewards)
        site_lo, site_hi = _minmax([_as_float(row["probe"].get("site_f1", 0.0)) for row in group_rows])
        for row in group_rows:
            probe = row["probe"]
            row["teacher_reward_sum"] = _as_float(probe.get("teacher_reward_sum", 0.0))
            row["site_f1"] = _as_float(probe.get("site_f1", 0.0))
            row["reward_norm"] = _norm(row["teacher_reward_sum"], reward_lo, reward_hi)
            row["site_norm"] = _norm(row["site_f1"], site_lo, site_hi)
            row["teacher_is_noop"] = bool(probe.get("teacher_is_noop", False))


def _metric_at_budget(row: dict[str, Any], budget: int) -> dict[str, float]:
    curve = row.get("pr_curve", {})
    if str(budget) in curve:
        raw = curve[str(budget)]
    elif budget in curve:
        raw = curve[budget]
    else:
        budgets = [int(key) for key in curve.keys()]
        budget = min(budgets, key=lambda item: abs(item - int(budget)))
        raw = curve[str(budget)]
    metrics = {key: _as_float(value) for key, value in raw.items()}
    metrics["budget"] = float(budget)
    metrics["teacher_reward_sum"] = _as_float(row.get("teacher_reward_sum", 0.0))
    metrics["site_f1"] = _as_float(row.get("site_f1", 0.0))
    metrics["reward_norm"] = _as_float(row.get("reward_norm", 0.0))
    metrics["site_norm"] = _as_float(row.get("site_norm", 0.0))
    metrics["non_noop"] = 0.0 if bool(row.get("teacher_is_noop", False)) else 1.0
    selected_count = max(_as_float(metrics.get("selected_pair_count", budget)), 1.0)
    true_count = max(_as_float(row.get("true_pair_count", metrics.get("true_pair_count", 0.0))), 0.0)
    metrics["count_efficiency"] = float(min(true_count / selected_count, 1.0))
    metrics["compactness"] = float(1.0 / selected_count)
    metrics["balanced_pair_endpoint_f1"] = 0.5 * _as_float(metrics.get("pair_f1", 0.0)) + 0.5 * _as_float(
        metrics.get("endpoint_f1", 0.0)
    )
    metrics["endpoint_site_f1"] = 0.5 * _as_float(metrics.get("endpoint_f1", 0.0)) + 0.5 * metrics["site_f1"]
    return metrics


def _target_values(metrics: dict[str, float]) -> dict[str, float]:
    pair_precision = _as_float(metrics.get("pair_precision", 0.0))
    pair_recall = _as_float(metrics.get("pair_recall", 0.0))
    pair_f1 = _as_float(metrics.get("pair_f1", 0.0))
    endpoint_f1 = _as_float(metrics.get("endpoint_f1", 0.0))
    site_f1 = _as_float(metrics.get("site_f1", 0.0))
    reward_norm = _as_float(metrics.get("reward_norm", 0.0))
    count_eff = _as_float(metrics.get("count_efficiency", 0.0))
    compactness = _as_float(metrics.get("compactness", 0.0))
    non_noop = _as_float(metrics.get("non_noop", 1.0))
    pair_pruning = (
        0.30 * pair_precision
        + 0.35 * pair_f1
        + 0.25 * endpoint_f1
        + 0.05 * pair_recall
        + 0.05 * min(count_eff + compactness, 1.0)
    )
    energy_site = (
        0.45 * reward_norm
        + 0.35 * site_f1
        + 0.10 * endpoint_f1
        + 0.05 * pair_recall
        + 0.05 * non_noop
    )
    return {
        "v120_energy_site_quality": float(energy_site),
        "v120_pair_pruning_quality": float(pair_pruning),
        "v120_joint_balanced": float(0.50 * energy_site + 0.50 * pair_pruning),
        "v120_energy_site_pair": float(
            0.35 * reward_norm + 0.25 * site_f1 + 0.15 * pair_f1 + 0.15 * endpoint_f1 + 0.10 * pair_precision
        ),
        "v120_precision_first": float(
            0.25 * reward_norm + 0.20 * site_f1 + 0.25 * pair_precision + 0.20 * pair_f1 + 0.10 * endpoint_f1
        ),
    }


def _candidate_budget_features(row: dict[str, Any], budget: int) -> list[float]:
    base = row.get("features", [])
    if not isinstance(base, list):
        base = []
    return v118._metric_features([_as_float(value) for value in base], int(budget), _as_int(row.get("pair_count", 0)))


def _candidate_budget_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        for budget in v118.BUDGETS:
            metrics = _metric_at_budget(row, int(budget))
            targets = _target_values(metrics)
            out.append(
                {
                    "key": _candidate_key(row),
                    "group": _group_key(row),
                    "source_name": row.get("source_name"),
                    "segment_index": _as_int(row.get("segment_index", 0)),
                    "candidate_index": _as_int(row.get("candidate_index", 0)),
                    "segment_k": _as_int(row.get("segment_k", 0)),
                    "selected_by_planner": bool(row.get("selected_by_planner", False)),
                    "budget": int(budget),
                    "features": _candidate_budget_features(row, int(budget)),
                    "metrics": metrics,
                    "targets": targets,
                }
            )
    return out


def _summarize_choices(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    metrics = [row["metrics"] for row in rows]
    targets = [row["targets"] for row in rows]
    out = {
        "candidate_count": float(len(rows)),
        "avg_budget": _mean([_as_float(row.get("budget", 0.0)) for row in rows]),
        "avg_site_f1": _mean([_as_float(m.get("site_f1", 0.0)) for m in metrics]),
        "avg_teacher_reward_sum": _mean([_as_float(m.get("teacher_reward_sum", 0.0)) for m in metrics]),
        "avg_reward_norm": _mean([_as_float(m.get("reward_norm", 0.0)) for m in metrics]),
        "avg_pair_precision": _mean([_as_float(m.get("pair_precision", 0.0)) for m in metrics]),
        "avg_pair_recall": _mean([_as_float(m.get("pair_recall", 0.0)) for m in metrics]),
        "avg_pair_f1": _mean([_as_float(m.get("pair_f1", 0.0)) for m in metrics]),
        "avg_endpoint_f1": _mean([_as_float(m.get("endpoint_f1", 0.0)) for m in metrics]),
        "avg_balanced_pair_endpoint_f1": _mean(
            [_as_float(m.get("balanced_pair_endpoint_f1", 0.0)) for m in metrics]
        ),
        "avg_count_efficiency": _mean([_as_float(m.get("count_efficiency", 0.0)) for m in metrics]),
        "avg_selected_pair_count": _mean([_as_float(m.get("selected_pair_count", 0.0)) for m in metrics]),
    }
    for name in OBJECTIVES:
        out[f"avg_{name}"] = _mean([_as_float(t.get(name, 0.0)) for t in targets])
    return out


def _source_hist(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("source_name", "")) for row in rows))


def _pick_selected_fixed(rows: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    selected = [row for row in rows if bool(row.get("selected_by_planner", False)) and int(row["budget"]) == budget]
    if selected:
        return selected[0]
    matching = [row for row in rows if int(row["budget"]) == budget]
    return matching[0] if matching else rows[0]


def _pick_oracle(rows: list[dict[str, Any]], objective: str) -> dict[str, Any]:
    return max(rows, key=lambda row: (_as_float(row["targets"].get(objective, 0.0)), -_as_float(row.get("budget", 0.0))))


def _fit_objective_selector(
    train_rows: list[dict[str, Any]],
    *,
    objective: str,
    ridge_l2: float,
) -> dict[str, Any]:
    xs = [row["features"] for row in train_rows]
    ys = [_as_float(row["targets"].get(objective, 0.0)) for row in train_rows]
    return v119._fit_ridge_regressor(xs, ys, l2=ridge_l2)


def _pick_predicted(rows: list[dict[str, Any]], model: dict[str, Any]) -> tuple[dict[str, Any], float]:
    preds = v119._predict_ridge(model, [row["features"] for row in rows])
    best_idx = max(range(len(rows)), key=lambda idx: (preds[idx], -_as_float(rows[idx].get("budget", 0.0))))
    return rows[best_idx], float(preds[best_idx])


def _fold_eval(rows: list[dict[str, Any]], *, ridge_l2: float) -> dict[str, Any]:
    groups = sorted({int(row["group"]) for row in rows})
    by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[int(row["group"])].append(row)

    baselines: dict[str, list[dict[str, Any]]] = defaultdict(list)
    policy_picks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    oracle_picks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    previews: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for group in groups:
        train_rows = [row for row in rows if int(row["group"]) != group]
        val_rows = by_group[group]
        baselines["selected_fixed_top16"].append(_pick_selected_fixed(val_rows, 16))
        baselines["selected_fixed_top32"].append(_pick_selected_fixed(val_rows, 32))
        baselines["selected_fixed_top64"].append(_pick_selected_fixed(val_rows, 64))
        for objective in OBJECTIVES:
            model = _fit_objective_selector(train_rows, objective=objective, ridge_l2=ridge_l2)
            pick, pred = _pick_predicted(val_rows, model)
            policy_picks[objective].append(pick)
            oracle_picks[objective].append(_pick_oracle(val_rows, objective))
            previews[objective].append(
                {
                    "group": int(group),
                    "source_name": str(pick.get("source_name", "")),
                    "segment_k": int(_as_int(pick.get("segment_k", 0))),
                    "candidate_index": int(_as_int(pick.get("candidate_index", 0))),
                    "budget": int(_as_int(pick.get("budget", 0))),
                    "predicted_score": float(pred),
                    "target": float(_as_float(pick["targets"].get(objective, 0.0))),
                    "site_f1": float(_as_float(pick["metrics"].get("site_f1", 0.0))),
                    "teacher_reward_sum": float(_as_float(pick["metrics"].get("teacher_reward_sum", 0.0))),
                    "pair_precision": float(_as_float(pick["metrics"].get("pair_precision", 0.0))),
                    "pair_f1": float(_as_float(pick["metrics"].get("pair_f1", 0.0))),
                    "endpoint_f1": float(_as_float(pick["metrics"].get("endpoint_f1", 0.0))),
                    "selected_pair_count": float(_as_float(pick["metrics"].get("selected_pair_count", 0.0))),
                }
            )

    selected_fixed = baselines["selected_fixed_top32"]
    policies: dict[str, dict[str, Any]] = {}
    for name, picks in sorted(baselines.items()):
        policies[name] = {
            "summary": _summarize_choices(picks),
            "source_histogram": _source_hist(picks),
        }
    for name, picks in sorted(policy_picks.items()):
        beats = sum(
            1
            for pick, base in zip(picks, selected_fixed)
            if _as_float(pick["targets"].get("v120_joint_balanced", 0.0))
            > _as_float(base["targets"].get("v120_joint_balanced", 0.0)) + 1e-12
        )
        policies[f"loo_reg_{name}"] = {
            "summary": _summarize_choices(picks),
            "source_histogram": _source_hist(picks),
            "folds_beating_selected_fixed_top32_joint": int(beats),
            "folds_total": int(len(picks)),
            "preview": previews[name][:20],
        }
    for name, picks in sorted(oracle_picks.items()):
        policies[f"oracle_{name}"] = {
            "summary": _summarize_choices(picks),
            "source_histogram": _source_hist(picks),
        }
    return {
        "fold_count": int(len(groups)),
        "policies": policies,
    }


def _best_non_oracle(policies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, item in policies.items():
        if name.startswith("oracle_"):
            continue
        summary = item.get("summary", {})
        rows.append(
            {
                "name": name,
                "avg_v120_joint_balanced": _as_float(summary.get("avg_v120_joint_balanced", 0.0)),
                "avg_site_f1": _as_float(summary.get("avg_site_f1", 0.0)),
                "avg_teacher_reward_sum": _as_float(summary.get("avg_teacher_reward_sum", 0.0)),
                "avg_pair_precision": _as_float(summary.get("avg_pair_precision", 0.0)),
                "avg_pair_f1": _as_float(summary.get("avg_pair_f1", 0.0)),
                "avg_endpoint_f1": _as_float(summary.get("avg_endpoint_f1", 0.0)),
                "avg_selected_pair_count": _as_float(summary.get("avg_selected_pair_count", 0.0)),
            }
        )
    return max(rows, key=lambda row: (row["avg_v120_joint_balanced"], row["avg_site_f1"])) if rows else {}


def _simultaneous_improvements(policies: dict[str, dict[str, Any]], baseline: dict[str, float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    base_site = _as_float(baseline.get("avg_site_f1", 0.0))
    base_pair = _as_float(baseline.get("avg_pair_f1", 0.0))
    base_endpoint = _as_float(baseline.get("avg_endpoint_f1", 0.0))
    for name, item in sorted(policies.items()):
        if name.startswith("oracle_") or name.startswith("selected_fixed_"):
            continue
        summary = item.get("summary", {})
        site = _as_float(summary.get("avg_site_f1", 0.0))
        pair = _as_float(summary.get("avg_pair_f1", 0.0))
        endpoint = _as_float(summary.get("avg_endpoint_f1", 0.0))
        if site > base_site + 1e-12 and pair > base_pair + 1e-12:
            out.append(
                {
                    "name": name,
                    "avg_site_f1": site,
                    "avg_pair_f1": pair,
                    "avg_endpoint_f1": endpoint,
                    "avg_selected_pair_count": _as_float(summary.get("avg_selected_pair_count", 0.0)),
                    "improves_endpoint_too": endpoint > base_endpoint + 1e-12,
                }
            )
    return out


def _write_target_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    compact_rows: list[dict[str, Any]] = []
    for row in rows:
        compact_rows.append(
            {
                "source_name": row["source_name"],
                "segment_index": row["segment_index"],
                "candidate_index": row["candidate_index"],
                "segment_k": row["segment_k"],
                "selected_by_planner": row["selected_by_planner"],
                "budget": row["budget"],
                "metrics": row["metrics"],
                "targets": row["targets"],
            }
        )
    _write_jsonl(output_path, compact_rows)


def run(
    *,
    v119_candidate_jsonl: Path,
    v115_candidate_jsonl: Path,
    output_dir: Path,
    ridge_l2: float,
) -> dict[str, Any]:
    v119_rows = _load_jsonl(v119_candidate_jsonl)
    v115_rows = _load_jsonl(v115_candidate_jsonl)
    joined = _join_candidate_rows(v119_rows, v115_rows)
    if not joined:
        raise RuntimeError("no joined v119/v115 candidate rows")
    _attach_group_targets(joined)
    expanded_rows = _candidate_budget_rows(joined)
    grouped = _fold_eval(expanded_rows, ridge_l2=ridge_l2)
    policies = grouped["policies"]
    selected_fixed = policies["selected_fixed_top32"]["summary"]
    best = _best_non_oracle(policies)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_rows_path = output_dir / "candidate_budget_targets_v120.jsonl"
    _write_target_rows(expanded_rows, target_rows_path)
    summary = {
        "mode": "v120 pure-python candidate-quality + pair-pruning selector diagnostic",
        "input_files": {
            "v119_candidate_jsonl": str(v119_candidate_jsonl),
            "v115_candidate_jsonl": str(v115_candidate_jsonl),
        },
        "candidate_count": int(len(joined)),
        "candidate_budget_row_count": int(len(expanded_rows)),
        "group_count": int(grouped["fold_count"]),
        "ridge_l2": float(ridge_l2),
        "target_definition": {
            "energy_site_quality": (
                "0.45 reward_norm + 0.35 site_f1 + 0.10 endpoint_f1 + "
                "0.05 pair_recall + 0.05 non_noop"
            ),
            "pair_pruning_quality": (
                "0.30 pair_precision + 0.35 pair_f1 + 0.25 endpoint_f1 + "
                "0.05 pair_recall + 0.05 capped_count_compactness"
            ),
            "v120_joint_balanced": "0.50 energy_site_quality + 0.50 pair_pruning_quality",
        },
        "available_risk_fields": {
            "teacher_is_noop": any("teacher_is_noop" in row["probe"] for row in joined),
            "model_noop_risk": any("model_noop_risk" in row["probe"] for row in joined),
            "model_expected_tau": any("model_expected_tau" in row["probe"] for row in joined),
        },
        "selected_fixed_top32": selected_fixed,
        "best_non_oracle": best,
        "non_oracle_policies_improving_site_and_pair_f1": _simultaneous_improvements(policies, selected_fixed),
        "best_non_oracle_minus_selected_fixed_top32": {
            "avg_v120_joint_balanced": _as_float(best.get("avg_v120_joint_balanced", 0.0))
            - _as_float(selected_fixed.get("avg_v120_joint_balanced", 0.0)),
            "avg_site_f1": _as_float(best.get("avg_site_f1", 0.0))
            - _as_float(selected_fixed.get("avg_site_f1", 0.0)),
            "avg_pair_f1": _as_float(best.get("avg_pair_f1", 0.0))
            - _as_float(selected_fixed.get("avg_pair_f1", 0.0)),
            "avg_pair_precision": _as_float(best.get("avg_pair_precision", 0.0))
            - _as_float(selected_fixed.get("avg_pair_precision", 0.0)),
            "avg_endpoint_f1": _as_float(best.get("avg_endpoint_f1", 0.0))
            - _as_float(selected_fixed.get("avg_endpoint_f1", 0.0)),
        },
        "grouped_eval": grouped,
        "output_files": {
            "summary": str(output_dir / "stage_summary.json"),
            "candidate_budget_targets": str(target_rows_path),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    args = parser.parse_args()
    result = run(
        v119_candidate_jsonl=args.v119_candidate_jsonl,
        v115_candidate_jsonl=args.v115_candidate_jsonl,
        output_dir=args.output_dir,
        ridge_l2=float(args.ridge_l2),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
