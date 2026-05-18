#!/usr/bin/env python3
"""Pure-Python v122 constrained/Pareto candidate selector diagnostic.

v121 showed that planner-visible features can improve either the
site/pair-support branch or the reward/site branch, but not all objectives at
once.  This read-only diagnostic keeps the same candidate-budget table and
tests whether explicit constraints or Pareto selection can synchronize:

* teacher reward / energy progress
* selected terminal site overlap
* vacancy-pair precision and F1
* endpoint F1 / support compactness

It is intentionally torch-free and does not modify checkpoints, caches, or
planner behavior.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import diagnose_candidate_feature_gap_v121 as v121
import diagnose_candidate_pruning_selector_v120 as v120
import diagnose_vacancy_pair_selector_v119 as v119


PREDICTED_TARGETS = (
    "teacher_reward_sum",
    "reward_norm",
    "site_f1",
    "pair_precision",
    "pair_f1",
    "endpoint_f1",
    "pair_recall",
)

POLICY_NAMES = (
    "pred_pareto_balanced",
    "pred_reward_floor_pair_first",
    "pred_reward_site_floor_pair_first",
    "pred_pair_floor_energy_site",
    "pred_two_branch_floor_balanced",
    "pred_soft_penalty_balanced",
    "pred_strict_triple_floor",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    return v120._as_float(value, default)


def _mean(values: list[float]) -> float:
    return v120._mean(values)


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = _mean(values)
    return float(math.sqrt(_mean([(value - avg) ** 2 for value in values])))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return v120._load_jsonl(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _minmax(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return lo, lo + 1.0
    return lo, hi


def _norm(value: float, lo: float, hi: float) -> float:
    return float((value - lo) / max(hi - lo, 1e-12))


def _metric(row: dict[str, Any], name: str) -> float:
    return _as_float(row.get("metrics", {}).get(name, 0.0))


def _pred(row: dict[str, Any], name: str) -> float:
    return _as_float(row.get("predictions", {}).get(name, 0.0))


def _target_value(row: dict[str, Any], name: str) -> float:
    if name in row.get("metrics", {}):
        return _metric(row, name)
    if name in row.get("targets", {}):
        return _as_float(row["targets"].get(name, 0.0))
    return 0.0


def _source_hist(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("source_name", "")) for row in rows))


def _summarize_choices(rows: list[dict[str, Any]]) -> dict[str, float]:
    base = v120._summarize_choices(rows)
    base["avg_reward_site_pair_product"] = _mean(
        [
            _metric(row, "site_f1")
            * max(_metric(row, "pair_f1"), 0.0)
            * max(_metric(row, "reward_norm"), 0.0)
            for row in rows
        ]
    )
    base["avg_reward_minus_fixed_top32"] = _mean(
        [_metric(row, "teacher_reward_sum") - _as_float(row.get("baseline_metrics", {}).get("teacher_reward_sum", 0.0)) for row in rows]
    )
    return base


def _feature_rows(
    *,
    v119_candidate_jsonl: Path,
    v115_candidate_jsonl: Path,
    v104_candidate_jsonl: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v119_rows = v120._load_jsonl(v119_candidate_jsonl)
    v115_rows = v120._load_jsonl(v115_candidate_jsonl)
    joined = v120._join_candidate_rows(v119_rows, v115_rows)
    if not joined:
        raise RuntimeError("no joined v119/v115 candidate rows")
    v120._attach_group_targets(joined)
    expanded_rows = v120._candidate_budget_rows(joined)
    v104_features, v104_summary = v121._load_v104_features(v104_candidate_jsonl)
    rows, join_summary = v121._with_feature_set(
        expanded_rows,
        feature_name="combined_v120_plus_v104",
        v104_features=v104_features,
    )
    return rows, {"v104_summary": v104_summary, "join_summary": join_summary}


def _fit_models(train_rows: list[dict[str, Any]], *, ridge_l2: float) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    xs = [row["features"] for row in train_rows]
    for target in PREDICTED_TARGETS:
        ys = [_target_value(row, target) for row in train_rows]
        models[target] = v119._fit_ridge_regressor(xs, ys, l2=ridge_l2)
    return models


def _attach_predictions(rows: list[dict[str, Any]], models: dict[str, dict[str, Any]]) -> None:
    xs = [row["features"] for row in rows]
    for target, model in models.items():
        preds = v119._predict_ridge(model, xs)
        for row, pred in zip(rows, preds):
            row.setdefault("predictions", {})[target] = float(pred)


def _selected_fixed(rows: list[dict[str, Any]], budget: int = 32) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if bool(row.get("selected_by_planner", False)) and int(row.get("budget", 0)) == int(budget)
    ]
    if selected:
        return selected[0]
    fallback = [row for row in rows if int(row.get("budget", 0)) == int(budget)]
    return fallback[0] if fallback else rows[0]


def _group_normalized_prediction_score(row: dict[str, Any], stats: dict[str, tuple[float, float]]) -> dict[str, float]:
    reward_raw = _norm(_pred(row, "teacher_reward_sum"), *stats["teacher_reward_sum"])
    reward_rel = _norm(_pred(row, "reward_norm"), *stats["reward_norm"])
    return {
        "reward": 0.5 * reward_raw + 0.5 * reward_rel,
        "site": _norm(_pred(row, "site_f1"), *stats["site_f1"]),
        "pair_precision": _norm(_pred(row, "pair_precision"), *stats["pair_precision"]),
        "pair_f1": _norm(_pred(row, "pair_f1"), *stats["pair_f1"]),
        "pair_recall": _norm(_pred(row, "pair_recall"), *stats["pair_recall"]),
        "endpoint": _norm(_pred(row, "endpoint_f1"), *stats["endpoint_f1"]),
    }


def _prediction_stats(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    return {name: _minmax([_pred(row, name) for row in rows]) for name in PREDICTED_TARGETS}


def _balanced_pred_score(row: dict[str, Any], stats: dict[str, tuple[float, float]]) -> float:
    values = _group_normalized_prediction_score(row, stats)
    return float(
        0.24 * values["reward"]
        + 0.22 * values["site"]
        + 0.22 * values["pair_f1"]
        + 0.18 * values["pair_precision"]
        + 0.14 * values["endpoint"]
    )


def _pair_first_pred_score(row: dict[str, Any], stats: dict[str, tuple[float, float]]) -> float:
    values = _group_normalized_prediction_score(row, stats)
    return float(
        0.30 * values["pair_precision"]
        + 0.30 * values["pair_f1"]
        + 0.20 * values["endpoint"]
        + 0.12 * values["site"]
        + 0.08 * values["reward"]
    )


def _energy_site_pred_score(row: dict[str, Any], stats: dict[str, tuple[float, float]]) -> float:
    values = _group_normalized_prediction_score(row, stats)
    return float(0.42 * values["reward"] + 0.35 * values["site"] + 0.13 * values["endpoint"] + 0.10 * values["pair_f1"])


def _filter_or_all(rows: list[dict[str, Any]], predicate) -> list[dict[str, Any]]:
    kept = [row for row in rows if predicate(row)]
    return kept if kept else rows


def _dominates(a: dict[str, float], b: dict[str, float], keys: tuple[str, ...]) -> bool:
    return all(a[key] >= b[key] - 1e-12 for key in keys) and any(a[key] > b[key] + 1e-12 for key in keys)


def _pareto_front(rows: list[dict[str, Any]], stats: dict[str, tuple[float, float]]) -> list[dict[str, Any]]:
    keys = ("reward", "site", "pair_precision", "pair_f1", "endpoint")
    values = {id(row): _group_normalized_prediction_score(row, stats) for row in rows}
    front: list[dict[str, Any]] = []
    for row in rows:
        row_values = values[id(row)]
        if not any(_dominates(values[id(other)], row_values, keys) for other in rows if other is not row):
            front.append(row)
    return front or rows


def _pick_by_score(rows: list[dict[str, Any]], score_fn) -> dict[str, Any]:
    return max(rows, key=lambda row: (score_fn(row), -_metric(row, "selected_pair_count")))


def _oracle_pareto(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("teacher_reward_sum", "site_f1", "pair_precision", "pair_f1", "endpoint_f1")
    stats = {key: _minmax([_metric(row, key) for row in rows]) for key in keys}

    def value(row: dict[str, Any]) -> dict[str, float]:
        return {key: _norm(_metric(row, key), *stats[key]) for key in keys}

    vals = {id(row): value(row) for row in rows}
    front = [
        row
        for row in rows
        if not any(_dominates(vals[id(other)], vals[id(row)], keys) for other in rows if other is not row)
    ]
    return _pick_by_score(
        front or rows,
        lambda row: (
            0.24 * vals[id(row)]["teacher_reward_sum"]
            + 0.22 * vals[id(row)]["site_f1"]
            + 0.22 * vals[id(row)]["pair_f1"]
            + 0.18 * vals[id(row)]["pair_precision"]
            + 0.14 * vals[id(row)]["endpoint_f1"]
        ),
    )


def _oracle_reward_pair_floor(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    reward_floor = _metric(baseline, "teacher_reward_sum")
    pair_floor = _metric(baseline, "pair_f1")
    precision_floor = _metric(baseline, "pair_precision")
    kept = _filter_or_all(
        rows,
        lambda row: _metric(row, "teacher_reward_sum") >= reward_floor - 1e-12
        and _metric(row, "pair_f1") >= pair_floor - 1e-12
        and _metric(row, "pair_precision") >= precision_floor - 1e-12,
    )
    return _pick_by_score(kept, lambda row: 0.35 * _metric(row, "site_f1") + 0.35 * _metric(row, "endpoint_f1") + 0.30 * _metric(row, "pair_f1"))


def _pick_policies(val_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    baseline = _selected_fixed(val_rows, 32)
    stats = _prediction_stats(val_rows)
    baseline_pred = baseline.get("predictions", {})
    reward_floor = 0.5 * _pred(baseline, "teacher_reward_sum") + 0.5 * _pred(baseline, "reward_norm")
    site_floor = _pred(baseline, "site_f1")
    pair_f1_floor = _pred(baseline, "pair_f1")
    pair_precision_floor = _pred(baseline, "pair_precision")

    def reward_value(row: dict[str, Any]) -> float:
        return 0.5 * _pred(row, "teacher_reward_sum") + 0.5 * _pred(row, "reward_norm")

    policies: dict[str, dict[str, Any]] = {
        "pred_pareto_balanced": _pick_by_score(
            _pareto_front(val_rows, stats),
            lambda row: _balanced_pred_score(row, stats),
        ),
        "pred_reward_floor_pair_first": _pick_by_score(
            _filter_or_all(val_rows, lambda row: reward_value(row) >= reward_floor - 1e-12),
            lambda row: _pair_first_pred_score(row, stats),
        ),
        "pred_reward_site_floor_pair_first": _pick_by_score(
            _filter_or_all(
                val_rows,
                lambda row: reward_value(row) >= reward_floor - 1e-12 and _pred(row, "site_f1") >= site_floor - 1e-12,
            ),
            lambda row: _pair_first_pred_score(row, stats),
        ),
        "pred_pair_floor_energy_site": _pick_by_score(
            _filter_or_all(
                val_rows,
                lambda row: _pred(row, "pair_f1") >= pair_f1_floor - 1e-12
                and _pred(row, "pair_precision") >= pair_precision_floor - 1e-12,
            ),
            lambda row: _energy_site_pred_score(row, stats),
        ),
        "pred_two_branch_floor_balanced": _pick_by_score(
            _filter_or_all(
                val_rows,
                lambda row: reward_value(row) >= reward_floor - 1e-12 and _pred(row, "pair_f1") >= pair_f1_floor - 1e-12,
            ),
            lambda row: _balanced_pred_score(row, stats),
        ),
        "pred_soft_penalty_balanced": _pick_by_score(
            val_rows,
            lambda row: _balanced_pred_score(row, stats)
            - 0.20 * max(0.0, reward_floor - reward_value(row))
            - 0.20 * max(0.0, pair_f1_floor - _pred(row, "pair_f1"))
            - 0.10 * max(0.0, pair_precision_floor - _pred(row, "pair_precision")),
        ),
        "pred_strict_triple_floor": _pick_by_score(
            _filter_or_all(
                val_rows,
                lambda row: reward_value(row) >= reward_floor - 1e-12
                and _pred(row, "site_f1") >= site_floor - 1e-12
                and _pred(row, "pair_f1") >= pair_f1_floor - 1e-12
                and _pred(row, "pair_precision") >= pair_precision_floor - 1e-12,
            ),
            lambda row: _balanced_pred_score(row, stats),
        ),
        "oracle_pareto_balanced": _oracle_pareto(val_rows),
        "oracle_reward_pair_floor": _oracle_reward_pair_floor(val_rows, baseline),
    }
    for row in policies.values():
        row["baseline_metrics"] = baseline.get("metrics", {})
        row["baseline_predictions"] = baseline_pred
    baseline["baseline_metrics"] = baseline.get("metrics", {})
    baseline["baseline_predictions"] = baseline_pred
    policies["selected_fixed_top32"] = baseline
    return policies


def _row_preview(row: dict[str, Any], group: int, policy: str) -> dict[str, Any]:
    return {
        "policy": policy,
        "group": int(group),
        "source_name": str(row.get("source_name", "")),
        "segment_k": int(v120._as_int(row.get("segment_k", 0))),
        "candidate_index": int(v120._as_int(row.get("candidate_index", 0))),
        "budget": int(v120._as_int(row.get("budget", 0))),
        "teacher_reward_sum": _metric(row, "teacher_reward_sum"),
        "site_f1": _metric(row, "site_f1"),
        "pair_precision": _metric(row, "pair_precision"),
        "pair_f1": _metric(row, "pair_f1"),
        "endpoint_f1": _metric(row, "endpoint_f1"),
        "selected_pair_count": _metric(row, "selected_pair_count"),
        "pred_teacher_reward_sum": _pred(row, "teacher_reward_sum"),
        "pred_site_f1": _pred(row, "site_f1"),
        "pred_pair_precision": _pred(row, "pair_precision"),
        "pred_pair_f1": _pred(row, "pair_f1"),
        "pred_endpoint_f1": _pred(row, "endpoint_f1"),
    }


def _fold_eval(rows: list[dict[str, Any]], *, ridge_l2: float) -> dict[str, Any]:
    by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[int(row["group"])].append(row)
    groups = sorted(by_group)

    picks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    previews: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model_quality: dict[str, list[float]] = defaultdict(list)

    for group in groups:
        train_rows = [row for row in rows if int(row["group"]) != group]
        val_rows = [dict(row) for row in by_group[group]]
        models = _fit_models(train_rows, ridge_l2=ridge_l2)
        _attach_predictions(val_rows, models)
        for target in PREDICTED_TARGETS:
            actual = [_target_value(row, target) for row in val_rows]
            pred = [_pred(row, target) for row in val_rows]
            model_quality[f"{target}_corr"].append(v121._pearson(actual, pred))
            model_quality[f"{target}_mae"].append(_mean([abs(a - b) for a, b in zip(actual, pred)]))
        group_policies = _pick_policies(val_rows)
        for policy, row in group_policies.items():
            picks[policy].append(row)
            previews[policy].append(_row_preview(row, group, policy))

    summaries: dict[str, dict[str, Any]] = {}
    baseline_picks = picks["selected_fixed_top32"]
    baseline = _summarize_choices(picks["selected_fixed_top32"])
    for policy, rows_for_policy in sorted(picks.items()):
        summary = _summarize_choices(rows_for_policy)
        folds_improving_all = 0
        if policy != "selected_fixed_top32":
            for row, base in zip(rows_for_policy, baseline_picks):
                if (
                    _metric(row, "teacher_reward_sum") >= _metric(base, "teacher_reward_sum") - 1e-12
                    and _metric(row, "site_f1") > _metric(base, "site_f1") + 1e-12
                    and _metric(row, "pair_precision") > _metric(base, "pair_precision") + 1e-12
                    and _metric(row, "pair_f1") > _metric(base, "pair_f1") + 1e-12
                ):
                    folds_improving_all += 1
        summaries[policy] = {
            "summary": summary,
            "source_histogram": _source_hist(rows_for_policy),
            "minus_selected_fixed_top32": {
                "avg_site_f1": _as_float(summary.get("avg_site_f1", 0.0)) - _as_float(baseline.get("avg_site_f1", 0.0)),
                "avg_teacher_reward_sum": _as_float(summary.get("avg_teacher_reward_sum", 0.0))
                - _as_float(baseline.get("avg_teacher_reward_sum", 0.0)),
                "avg_pair_precision": _as_float(summary.get("avg_pair_precision", 0.0))
                - _as_float(baseline.get("avg_pair_precision", 0.0)),
                "avg_pair_f1": _as_float(summary.get("avg_pair_f1", 0.0)) - _as_float(baseline.get("avg_pair_f1", 0.0)),
                "avg_endpoint_f1": _as_float(summary.get("avg_endpoint_f1", 0.0))
                - _as_float(baseline.get("avg_endpoint_f1", 0.0)),
            },
            "folds_improving_reward_site_pair_precision_f1": int(folds_improving_all),
            "folds_total": int(len(rows_for_policy)),
            "preview": previews[policy][:20],
        }

    return {
        "fold_count": int(len(groups)),
        "policies": summaries,
        "prediction_quality": {
            name: {"mean": _mean(values), "std": _std(values)}
            for name, values in sorted(model_quality.items())
        },
    }


def _find_sync_improvements(policies: dict[str, dict[str, Any]], baseline: dict[str, float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    base_reward = _as_float(baseline.get("avg_teacher_reward_sum", 0.0))
    base_site = _as_float(baseline.get("avg_site_f1", 0.0))
    base_precision = _as_float(baseline.get("avg_pair_precision", 0.0))
    base_pair = _as_float(baseline.get("avg_pair_f1", 0.0))
    for name, item in sorted(policies.items()):
        if name.startswith("oracle_") or name == "selected_fixed_top32":
            continue
        summary = item.get("summary", {})
        reward = _as_float(summary.get("avg_teacher_reward_sum", 0.0))
        site = _as_float(summary.get("avg_site_f1", 0.0))
        precision = _as_float(summary.get("avg_pair_precision", 0.0))
        pair = _as_float(summary.get("avg_pair_f1", 0.0))
        if reward >= base_reward - 1e-12 and site > base_site + 1e-12 and precision > base_precision + 1e-12 and pair > base_pair + 1e-12:
            out.append(
                {
                    "name": name,
                    "avg_teacher_reward_sum": reward,
                    "avg_site_f1": site,
                    "avg_pair_precision": precision,
                    "avg_pair_f1": pair,
                    "avg_endpoint_f1": _as_float(summary.get("avg_endpoint_f1", 0.0)),
                    "avg_selected_pair_count": _as_float(summary.get("avg_selected_pair_count", 0.0)),
                }
            )
    return out


def _best_non_oracle(policies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, item in sorted(policies.items()):
        if name.startswith("oracle_") or name == "selected_fixed_top32":
            continue
        summary = item.get("summary", {})
        rows.append(
            {
                "name": name,
                "avg_teacher_reward_sum": _as_float(summary.get("avg_teacher_reward_sum", 0.0)),
                "avg_site_f1": _as_float(summary.get("avg_site_f1", 0.0)),
                "avg_pair_precision": _as_float(summary.get("avg_pair_precision", 0.0)),
                "avg_pair_f1": _as_float(summary.get("avg_pair_f1", 0.0)),
                "avg_endpoint_f1": _as_float(summary.get("avg_endpoint_f1", 0.0)),
                "avg_selected_pair_count": _as_float(summary.get("avg_selected_pair_count", 0.0)),
            }
        )
    return max(
        rows,
        key=lambda row: (
            row["avg_teacher_reward_sum"] >= 0.0,
            row["avg_site_f1"] + row["avg_pair_f1"] + row["avg_pair_precision"],
            row["avg_teacher_reward_sum"],
        ),
    ) if rows else {}


def run(
    *,
    v119_candidate_jsonl: Path,
    v115_candidate_jsonl: Path,
    v104_candidate_jsonl: Path,
    output_dir: Path,
    ridge_l2: float,
) -> dict[str, Any]:
    rows, join_summary = _feature_rows(
        v119_candidate_jsonl=v119_candidate_jsonl,
        v115_candidate_jsonl=v115_candidate_jsonl,
        v104_candidate_jsonl=v104_candidate_jsonl,
    )
    grouped = _fold_eval(rows, ridge_l2=ridge_l2)
    policies = grouped["policies"]
    baseline = policies["selected_fixed_top32"]["summary"]
    sync = _find_sync_improvements(policies, baseline)
    best = _best_non_oracle(policies)

    output_dir.mkdir(parents=True, exist_ok=True)
    choice_rows: list[dict[str, Any]] = []
    for policy, item in sorted(policies.items()):
        for row in item.get("preview", []):
            choice_rows.append(row)
    choices_path = output_dir / "candidate_pareto_choices_v122.jsonl"
    _write_jsonl(choices_path, choice_rows)

    summary = {
        "mode": "v122 pure-python constrained/Pareto multi-objective candidate selector diagnostic",
        "input_files": {
            "v119_candidate_jsonl": str(v119_candidate_jsonl),
            "v115_candidate_jsonl": str(v115_candidate_jsonl),
            "v104_candidate_jsonl": str(v104_candidate_jsonl),
        },
        "candidate_budget_row_count": int(len(rows)),
        "candidate_count": int(len({tuple(row.get("key", ())) for row in rows})),
        "group_count": int(grouped["fold_count"]),
        "ridge_l2": float(ridge_l2),
        "join_summary": join_summary,
        "selected_fixed_top32": baseline,
        "best_non_oracle": best,
        "non_oracle_reward_site_pair_precision_f1_improvements": sync,
        "policy_eval": policies,
        "prediction_quality": grouped["prediction_quality"],
        "current_judgement": (
            "If no non-oracle policy improves reward, site F1, pair precision, and pair F1 "
            "over selected_fixed_top32, the blocker remains constrained multi-objective "
            "candidate modeling rather than a simple Pareto/floor post-selector."
        ),
        "output_files": {
            "summary": str(output_dir / "stage_summary.json"),
            "candidate_choices": str(choices_path),
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
