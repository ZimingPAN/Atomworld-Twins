#!/usr/bin/env python3
"""Pure-Python v118 grouped pruning-target diagnostic.

v117 showed that simple thresholds and fixed budgets do not generalize beyond
fixed top32.  This script uses the complete precision-recall curve from the
v115 pair distillation samples to test whether a small loader-level policy can
learn a support-count/pruning rule under grouped segment splits.

The script is intentionally torch-free and read-only.  It only writes diagnostic
JSON/JSONL artifacts that can later be used by a model-level pruning head.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


BUDGETS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512]
OBJECTIVES = ("pair_f1", "endpoint_f1", "balanced_f1")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return float(math.sqrt(_mean([(value - m) ** 2 for value in values])))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _candidate_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("source_name", "")),
        _as_int(row.get("segment_index", 0)),
        _as_int(row.get("candidate_index", 0)),
    )


def _group_key(row: dict[str, Any]) -> int:
    return _as_int(row.get("segment_index", row.get("fold_key", 0)))


def _pair_key(row: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(_as_int(value) for value in (row.get("source_position") or [])),
        tuple(_as_int(value) for value in (row.get("destination_position") or [])),
    )


def _pair_score(row: dict[str, Any], score_field: str) -> float:
    return _as_float(row.get(score_field, row.get("calibrated_interaction_score", 0.0)))


def _true_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _as_int(row.get("label", 0)) == 1]


def _support_metrics(selected: list[dict[str, Any]], true_rows: list[dict[str, Any]]) -> dict[str, float]:
    selected_pairs = {_pair_key(row) for row in selected}
    true_pairs = {_pair_key(row) for row in true_rows}
    pair_tp = len(selected_pairs & true_pairs)
    pair_precision = pair_tp / max(len(selected_pairs), 1)
    pair_recall = pair_tp / max(len(true_pairs), 1)

    selected_sites: set[tuple[int, ...]] = set()
    for row in selected:
        src, dst = _pair_key(row)
        selected_sites.add(src)
        selected_sites.add(dst)
    true_sites: set[tuple[int, ...]] = set()
    for row in true_rows:
        src, dst = _pair_key(row)
        true_sites.add(src)
        true_sites.add(dst)
    site_tp = len(selected_sites & true_sites)
    endpoint_precision = site_tp / max(len(selected_sites), 1)
    endpoint_recall = site_tp / max(len(true_sites), 1)
    pair_f1 = _f1(pair_precision, pair_recall)
    endpoint_f1 = _f1(endpoint_precision, endpoint_recall)
    return {
        "selected_pair_count": float(len(selected_pairs)),
        "true_pair_count": float(len(true_pairs)),
        "pair_precision": float(pair_precision),
        "pair_recall": float(pair_recall),
        "pair_f1": pair_f1,
        "endpoint_precision": float(endpoint_precision),
        "endpoint_recall": float(endpoint_recall),
        "endpoint_f1": endpoint_f1,
        "balanced_f1": 0.5 * pair_f1 + 0.5 * endpoint_f1,
    }


def _ranked(rows: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _pair_score(row, score_field), reverse=True)


def _summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row.keys()})
    return {key: _mean([_as_float(row.get(key, 0.0)) for row in rows]) for key in keys}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(max(int(round(q * (len(values) - 1))), 0), len(values) - 1)
    return float(values[idx])


def _budget_metrics(rows: list[dict[str, Any]], score_field: str) -> dict[int, dict[str, float]]:
    ranked = _ranked(rows, score_field)
    true = _true_rows(rows)
    return {
        budget: _support_metrics(ranked[: max(1, min(int(budget), len(ranked)))], true)
        for budget in BUDGETS
    }


def _objective_value(metrics: dict[str, float], objective: str) -> float:
    if objective == "balanced_f1":
        return 0.5 * _as_float(metrics.get("pair_f1", 0.0)) + 0.5 * _as_float(metrics.get("endpoint_f1", 0.0))
    return _as_float(metrics.get(objective, 0.0))


def _best_budget(curve: dict[int, dict[str, float]], objective: str) -> int:
    return max(
        BUDGETS,
        key=lambda budget: (
            _objective_value(curve[budget], objective),
            _as_float(curve[budget].get("pair_recall", 0.0)),
            -_as_float(curve[budget].get("selected_pair_count", 0.0)),
        ),
    )


def _curve_features(candidate: dict[str, Any], rows: list[dict[str, Any]], score_field: str) -> list[float]:
    scores = sorted([_pair_score(row, score_field) for row in rows], reverse=True)
    if not scores:
        scores = [0.0]
    def score_at(rank: int) -> float:
        idx = min(max(rank - 1, 0), len(scores) - 1)
        return float(scores[idx])
    source_name = str(candidate.get("source_name", ""))
    candidate_index = _as_int(candidate.get("candidate_index", 0))
    segment_k = _as_int(candidate.get("segment_k", 0))
    max_score = score_at(1)
    return [
        1.0,
        float(len(rows)),
        float(segment_k),
        float(candidate_index),
        1.0 if bool(candidate.get("selected_by_planner", False)) else 0.0,
        1.0 if "vacancy" in source_name else 0.0,
        1.0 if "energy" in source_name else 0.0,
        1.0 if "factorized" in source_name else 0.0,
        max_score,
        score_at(2),
        score_at(4),
        score_at(8),
        score_at(16),
        score_at(32),
        score_at(64),
        score_at(128),
        _percentile(scores, 0.50),
        _percentile(scores, 0.75),
        _percentile(scores, 0.90),
        _percentile(scores, 0.95),
        _percentile(scores, 0.99),
        _mean(scores),
        _std(scores),
        max_score - score_at(4),
        max_score - score_at(8),
        max_score - score_at(16),
        max_score - score_at(32),
        max_score - score_at(64),
        max_score - score_at(128),
    ]


def _metric_features(base_features: list[float], budget: int, rows_count: int) -> list[float]:
    frac = float(budget / max(rows_count, 1))
    return base_features + [
        math.log(float(budget)),
        float(budget),
        frac,
        math.sqrt(float(budget)),
        1.0 / max(float(budget), 1.0),
    ]


def _fit_normalizer(xs: list[list[float]]) -> tuple[list[float], list[float]]:
    if not xs:
        return [], []
    dim = len(xs[0])
    mean = [_mean([row[idx] for row in xs]) for idx in range(dim)]
    std = []
    for idx in range(dim):
        std.append(max(_std([row[idx] for row in xs]), 1e-6))
    return mean, std


def _normalize(row: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [(row[idx] - mean[idx]) / std[idx] for idx in range(len(row))]


def _dot(weights: list[float], row: list[float]) -> float:
    return float(sum(weight * value for weight, value in zip(weights, row)))


def _train_regressor(
    xs: list[list[float]],
    ys: list[float],
    *,
    epochs: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    if not xs:
        raise RuntimeError("empty training features")
    rng = random.Random(seed)
    mean, std = _fit_normalizer(xs)
    xsn = [_normalize(row, mean, std) + [1.0] for row in xs]
    y_mean = _mean(ys)
    y_std = max(_std(ys), 1e-6)
    ysn = [(value - y_mean) / y_std for value in ys]
    weights = [0.0 for _ in range(len(xsn[0]))]
    indices = list(range(len(xsn)))
    for _epoch in range(max(1, int(epochs))):
        rng.shuffle(indices)
        for idx in indices:
            pred = _dot(weights, xsn[idx])
            err = pred - ysn[idx]
            for feat_idx, value in enumerate(xsn[idx]):
                weights[feat_idx] -= lr * err * value
    return {
        "weights": weights,
        "mean": mean,
        "std": std,
        "target_mean": y_mean,
        "target_std": y_std,
    }


def _predict(model: dict[str, Any], xs: list[list[float]]) -> list[float]:
    out: list[float] = []
    for row in xs:
        x = _normalize(row, model["mean"], model["std"]) + [1.0]
        out.append(_dot(model["weights"], x) * float(model["target_std"]) + float(model["target_mean"]))
    return out


def _candidate_records(
    candidate_rows: list[dict[str, Any]],
    pair_rows_by_key: dict[tuple[str, int, int], list[dict[str, Any]]],
    score_field: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        key = _candidate_key(candidate)
        rows = pair_rows_by_key.get(key, [])
        if not rows:
            continue
        curve = _budget_metrics(rows, score_field)
        base_features = _curve_features(candidate, rows, score_field)
        best = {objective: _best_budget(curve, objective) for objective in OBJECTIVES}
        records.append(
            {
                "key": key,
                "group": _group_key(candidate),
                "source_name": candidate.get("source_name"),
                "segment_index": _as_int(candidate.get("segment_index", 0)),
                "candidate_index": _as_int(candidate.get("candidate_index", 0)),
                "segment_k": _as_int(candidate.get("segment_k", 0)),
                "selected_by_planner": bool(candidate.get("selected_by_planner", False)),
                "rows_count": int(len(rows)),
                "base_features": base_features,
                "curve": curve,
                "best_budget": best,
                "true_pair_count": int(curve[BUDGETS[-1]].get("true_pair_count", 0.0)),
            }
        )
    return records


def _evaluate_budget_policy(records: list[dict[str, Any]], budgets: dict[tuple[str, int, int], int]) -> dict[str, float]:
    metrics: list[dict[str, float]] = []
    for record in records:
        budget = budgets.get(record["key"], 32)
        if budget not in record["curve"]:
            budget = min(BUDGETS, key=lambda item: abs(item - budget))
        item = dict(record["curve"][budget])
        item["candidate_count"] = 1.0
        metrics.append(item)
    return _summarize(metrics)


def _fixed_budget_policy(records: list[dict[str, Any]], budget: int) -> dict[tuple[str, int, int], int]:
    return {record["key"]: int(budget) for record in records}


def _oracle_policy(records: list[dict[str, Any]], objective: str) -> dict[tuple[str, int, int], int]:
    return {record["key"]: int(record["best_budget"][objective]) for record in records}


def _curve_regression_policy(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    *,
    objective: str,
    epochs: int,
    lr: float,
    seed: int,
) -> dict[tuple[str, int, int], int]:
    xs: list[list[float]] = []
    ys: list[float] = []
    for record in train_records:
        for budget in BUDGETS:
            xs.append(_metric_features(record["base_features"], budget, int(record["rows_count"])))
            ys.append(_objective_value(record["curve"][budget], objective))
    model = _train_regressor(xs, ys, epochs=epochs, lr=lr, seed=seed)
    out: dict[tuple[str, int, int], int] = {}
    for record in eval_records:
        features = [_metric_features(record["base_features"], budget, int(record["rows_count"])) for budget in BUDGETS]
        preds = _predict(model, features)
        best_idx = max(
            range(len(BUDGETS)),
            key=lambda idx: (
                preds[idx],
                -BUDGETS[idx],
            ),
        )
        out[record["key"]] = BUDGETS[best_idx]
    return out


def _budget_regression_policy(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    *,
    objective: str,
    epochs: int,
    lr: float,
    seed: int,
) -> dict[tuple[str, int, int], int]:
    xs = [record["base_features"] for record in train_records]
    ys = [math.log(float(record["best_budget"][objective])) for record in train_records]
    model = _train_regressor(xs, ys, epochs=epochs, lr=lr, seed=seed)
    preds = _predict(model, [record["base_features"] for record in eval_records])
    out: dict[tuple[str, int, int], int] = {}
    for record, pred in zip(eval_records, preds):
        budget = min(BUDGETS, key=lambda item: abs(math.log(float(item)) - pred))
        out[record["key"]] = budget
    return out


def _grouped_eval(
    records: list[dict[str, Any]],
    *,
    epochs: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    groups = sorted({record["group"] for record in records})
    fold_rows: list[dict[str, Any]] = []
    policy_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    selected_policy_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    for fold_idx, group in enumerate(groups):
        train_records = [record for record in records if record["group"] != group]
        val_records = [record for record in records if record["group"] == group]
        selected_val_records = [record for record in val_records if bool(record.get("selected_by_planner", False))]
        policies: dict[str, dict[tuple[str, int, int], int]] = {
            "fixed_top16": _fixed_budget_policy(val_records, 16),
            "fixed_top24": _fixed_budget_policy(val_records, 24),
            "fixed_top32": _fixed_budget_policy(val_records, 32),
            "fixed_top48": _fixed_budget_policy(val_records, 48),
            "oracle_pair_f1": _oracle_policy(val_records, "pair_f1"),
            "oracle_endpoint_f1": _oracle_policy(val_records, "endpoint_f1"),
            "oracle_balanced_f1": _oracle_policy(val_records, "balanced_f1"),
        }
        for objective in OBJECTIVES:
            policies[f"curve_reg_{objective}"] = _curve_regression_policy(
                train_records,
                val_records,
                objective=objective,
                epochs=epochs,
                lr=lr,
                seed=seed + 19 * fold_idx,
            )
            policies[f"budget_reg_{objective}"] = _budget_regression_policy(
                train_records,
                val_records,
                objective=objective,
                epochs=epochs,
                lr=lr,
                seed=seed + 101 + 19 * fold_idx,
            )
        for name, budget_policy in policies.items():
            metrics = _evaluate_budget_policy(val_records, budget_policy)
            selected_metrics = _evaluate_budget_policy(selected_val_records, budget_policy) if selected_val_records else {}
            policy_metrics[name].append(metrics)
            if selected_metrics:
                selected_policy_metrics[name].append(selected_metrics)
        fold_rows.append(
            {
                "group": int(group),
                "val_candidate_count": int(len(val_records)),
                "selected_val_candidate_count": int(len(selected_val_records)),
                "fixed_top32": policy_metrics["fixed_top32"][-1],
                "curve_reg_balanced_f1": policy_metrics["curve_reg_balanced_f1"][-1],
                "oracle_balanced_f1": policy_metrics["oracle_balanced_f1"][-1],
            }
        )
    summary: dict[str, Any] = {
        "fold_count": int(len(groups)),
        "policies": {},
        "selected_policies": {},
        "folds": fold_rows,
    }
    fixed_balanced_by_fold = policy_metrics["fixed_top32"]
    for name, rows in sorted(policy_metrics.items()):
        aggregate = _summarize(rows)
        beats = sum(
            1
            for row, baseline in zip(rows, fixed_balanced_by_fold)
            if _objective_value(row, "balanced_f1") > _objective_value(baseline, "balanced_f1") + 1e-12
        )
        aggregate["folds_beating_fixed_top32_balanced"] = float(beats)
        aggregate["folds_total"] = float(len(rows))
        summary["policies"][name] = aggregate
    for name, rows in sorted(selected_policy_metrics.items()):
        summary["selected_policies"][name] = _summarize(rows)
    return summary


def _candidate_target_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        curve = {
            str(budget): {
                key: _as_float(value)
                for key, value in record["curve"][budget].items()
            }
            for budget in BUDGETS
        }
        rows.append(
            {
                "source_name": record["source_name"],
                "segment_index": record["segment_index"],
                "candidate_index": record["candidate_index"],
                "segment_k": record["segment_k"],
                "selected_by_planner": record["selected_by_planner"],
                "pair_count": record["rows_count"],
                "true_pair_count": record["true_pair_count"],
                "best_budget": record["best_budget"],
                "pr_curve": curve,
                "features": record["base_features"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-field", default="calibrated_interaction_score")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = _load_jsonl(args.pair_jsonl)
    candidate_rows = _load_jsonl(args.candidate_jsonl)
    if not pair_rows or not candidate_rows:
        raise RuntimeError("non-empty pair/candidate JSONL inputs are required")

    pair_rows_by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        pair_rows_by_key[_candidate_key(row)].append(row)
    records = _candidate_records(candidate_rows, pair_rows_by_key, args.score_field)
    if not records:
        raise RuntimeError("no candidate records built from input JSONL")

    grouped = _grouped_eval(records, epochs=int(args.epochs), lr=float(args.lr), seed=int(args.seed))
    fixed = grouped["policies"]["fixed_top32"]
    ranked_policies = sorted(
        [
            {
                "name": name,
                "pair_f1": _as_float(metrics.get("pair_f1", 0.0)),
                "endpoint_f1": _as_float(metrics.get("endpoint_f1", 0.0)),
                "balanced_f1": _as_float(metrics.get("balanced_f1", 0.0)),
                "selected_pair_count": _as_float(metrics.get("selected_pair_count", 0.0)),
                "folds_beating_fixed_top32_balanced": _as_float(
                    metrics.get("folds_beating_fixed_top32_balanced", 0.0)
                ),
            }
            for name, metrics in grouped["policies"].items()
            if not name.startswith("oracle_")
        ],
        key=lambda row: (row["balanced_f1"], row["pair_f1"]),
        reverse=True,
    )
    best_non_oracle = ranked_policies[0] if ranked_policies else {}
    summary = {
        "mode": "v118 pure-python grouped precision-recall pruning target diagnostic",
        "score_field": args.score_field,
        "input_files": {
            "pair_jsonl": str(args.pair_jsonl),
            "candidate_jsonl": str(args.candidate_jsonl),
        },
        "pair_count": int(len(pair_rows)),
        "candidate_count": int(len(records)),
        "group_count": int(len({record["group"] for record in records})),
        "fixed_top32_grouped": fixed,
        "best_non_oracle_grouped": best_non_oracle,
        "best_non_oracle_minus_fixed_top32": {
            "pair_f1": _as_float(best_non_oracle.get("pair_f1", 0.0)) - _as_float(fixed.get("pair_f1", 0.0)),
            "endpoint_f1": _as_float(best_non_oracle.get("endpoint_f1", 0.0)) - _as_float(fixed.get("endpoint_f1", 0.0)),
            "balanced_f1": _as_float(best_non_oracle.get("balanced_f1", 0.0)) - _as_float(fixed.get("balanced_f1", 0.0)),
        },
        "grouped_eval": grouped,
        "best_budget_histogram": {
            objective: {
                str(budget): sum(1 for record in records if int(record["best_budget"][objective]) == budget)
                for budget in BUDGETS
            }
            for objective in OBJECTIVES
        },
        "output_files": {
            "summary": str(args.output_dir / "stage_summary.json"),
            "candidate_targets": str(args.output_dir / "candidate_pr_curve_targets_v118.jsonl"),
        },
    }
    (args.output_dir / "stage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _write_jsonl(args.output_dir / "candidate_pr_curve_targets_v118.jsonl", _candidate_target_rows(records))
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
