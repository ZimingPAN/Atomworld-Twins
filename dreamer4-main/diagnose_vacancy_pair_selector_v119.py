#!/usr/bin/env python3
"""Pure-Python v119 joint pair-score and pruning selector diagnostic.

v118 showed that a grouped PR-curve budget regressor can beat fixed top32 when
it uses the oracle-calibrated pair score.  This read-only diagnostic removes
that last shortcut: each leave-one-segment-out fold first distills a pair
compatibility scorer from train-fold v115 targets, then trains a PR-curve
support-count selector on the predicted pair scores, and finally evaluates the
held-out segment.  It is intentionally torch-free and does not modify any
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import diagnose_vacancy_pair_pruning_v118 as v118
import train_vacancy_pair_distill_v116 as v116


PRED_SCORE_FIELD = "v119_predicted_pair_score"


def _as_float(value: Any, default: float = 0.0) -> float:
    return v118._as_float(value, default)


def _as_int(value: Any, default: int = 0) -> int:
    return v118._as_int(value, default)


def _mean(values: list[float]) -> float:
    return v118._mean(values)


def _std(values: list[float]) -> float:
    return v118._std(values)


def _pearson(xs: list[float], ys: list[float]) -> float:
    return v116._pearson(xs, ys)


def _candidate_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return v118._candidate_key(row)


def _group_key(row: dict[str, Any]) -> int:
    return v118._group_key(row)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return v118._load_jsonl(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _pair_features(row: dict[str, Any]) -> list[float]:
    return v116._pair_features(row)


def _fit_normalizer(xs: list[list[float]]) -> tuple[list[float], list[float]]:
    if not xs:
        return [], []
    dim = len(xs[0])
    mean = [_mean([row[idx] for row in xs]) for idx in range(dim)]
    std = [max(_std([row[idx] for row in xs]), 1e-6) for idx in range(dim)]
    return mean, std


def _normalize(row: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [(row[idx] - mean[idx]) / std[idx] for idx in range(len(row))]


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    aug = [list(matrix[idx]) + [float(rhs[idx])] for idx in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            aug[pivot][col] = 1e-12
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        denom = aug[col][col]
        for item in range(col, n + 1):
            aug[col][item] /= denom
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) <= 0.0:
                continue
            for item in range(col, n + 1):
                aug[row][item] -= factor * aug[col][item]
    return [aug[row][n] for row in range(n)]


def _fit_ridge_regressor(
    xs: list[list[float]],
    ys: list[float],
    *,
    l2: float,
) -> dict[str, Any]:
    mean, std = _fit_normalizer(xs)
    xsn = [_normalize(row, mean, std) + [1.0] for row in xs]
    y_mean = _mean(ys)
    y_std = max(_std(ys), 1e-6)
    ysn = [(value - y_mean) / y_std for value in ys]
    dim = len(xsn[0])
    xtx = [[0.0 for _ in range(dim)] for _ in range(dim)]
    xty = [0.0 for _ in range(dim)]
    for row, target in zip(xsn, ysn):
        for i, xi in enumerate(row):
            xty[i] += xi * target
            for j, xj in enumerate(row):
                xtx[i][j] += xi * xj
    ridge = float(max(l2, 0.0))
    for idx in range(dim - 1):
        xtx[idx][idx] += ridge
    xtx[-1][-1] += ridge * 1e-6
    weights = _solve_linear_system(xtx, xty)
    return {
        "weights": weights,
        "mean": mean,
        "std": std,
        "target_mean": y_mean,
        "target_std": y_std,
        "l2": ridge,
    }


def _predict_ridge(model: dict[str, Any], xs: list[list[float]]) -> list[float]:
    out: list[float] = []
    weights = model["weights"]
    for row in xs:
        x = _normalize(row, model["mean"], model["std"]) + [1.0]
        out.append(
            sum(float(weight) * float(value) for weight, value in zip(weights, x))
            * float(model["target_std"])
            + float(model["target_mean"])
        )
    return out


def _split_pair_rows(
    pair_rows: list[dict[str, Any]],
    candidate_groups: dict[tuple[str, int, int], int],
    heldout_group: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for row in pair_rows:
        group = candidate_groups.get(_candidate_key(row), _group_key(row))
        if group == heldout_group:
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows


def _fit_pair_model(
    train_rows: list[dict[str, Any]],
    *,
    target_field: str,
    ridge_l2: float,
    max_train_rows: int,
    seed: int,
) -> dict[str, Any]:
    if max_train_rows > 0 and len(train_rows) > max_train_rows:
        train_rows = random.Random(seed).sample(train_rows, int(max_train_rows))
    xs = [_pair_features(row) for row in train_rows]
    ys = [_as_float(row.get(target_field, 0.0)) for row in train_rows]
    if not xs:
        raise RuntimeError("empty pair train fold")
    return _fit_ridge_regressor(xs, ys, l2=ridge_l2)


def _predict_pair_rows(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    target_field: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    xs = [_pair_features(row) for row in rows]
    targets = [_as_float(row.get(target_field, 0.0)) for row in rows]
    preds = _predict_ridge(model, xs) if xs else []
    out: list[dict[str, Any]] = []
    mse = 0.0
    for row, pred, target in zip(rows, preds, targets):
        item = dict(row)
        item[PRED_SCORE_FIELD] = float(pred)
        out.append(item)
        mse += (float(pred) - float(target)) ** 2
    metrics = {
        "pair_count": float(len(rows)),
        "corr": _pearson(preds, targets),
        "mse": float(mse / max(len(rows), 1)),
        "pred_mean": _mean([float(value) for value in preds]) if preds else 0.0,
        "pred_std": _std([float(value) for value in preds]) if preds else 0.0,
        "target_mean": _mean(targets) if targets else 0.0,
        "target_std": _std(targets) if targets else 0.0,
    }
    return out, metrics


def _records_for_candidates(
    candidate_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pair_rows_by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        pair_rows_by_key[_candidate_key(row)].append(row)
    return v118._candidate_records(candidate_rows, pair_rows_by_key, PRED_SCORE_FIELD)


def _policy_metrics(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    *,
    epochs: int,
    lr: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    policies: dict[str, dict[tuple[str, int, int], int]] = {
        "fixed_top16": v118._fixed_budget_policy(val_records, 16),
        "fixed_top24": v118._fixed_budget_policy(val_records, 24),
        "fixed_top32": v118._fixed_budget_policy(val_records, 32),
        "fixed_top48": v118._fixed_budget_policy(val_records, 48),
        "oracle_pair_f1": v118._oracle_policy(val_records, "pair_f1"),
        "oracle_endpoint_f1": v118._oracle_policy(val_records, "endpoint_f1"),
        "oracle_balanced_f1": v118._oracle_policy(val_records, "balanced_f1"),
    }
    for objective in v118.OBJECTIVES:
        policies[f"curve_reg_{objective}"] = v118._curve_regression_policy(
            train_records,
            val_records,
            objective=objective,
            epochs=epochs,
            lr=lr,
            seed=seed + 17,
        )
        policies[f"budget_reg_{objective}"] = v118._budget_regression_policy(
            train_records,
            val_records,
            objective=objective,
            epochs=epochs,
            lr=lr,
            seed=seed + 101,
        )
    return {
        name: v118._evaluate_budget_policy(val_records, budget_policy)
        for name, budget_policy in policies.items()
    }


def _aggregate_policy_rows(
    policy_rows: dict[str, list[dict[str, float]]],
    *,
    fixed_rows: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, rows in sorted(policy_rows.items()):
        aggregate = v118._summarize(rows)
        beats = sum(
            1
            for row, baseline in zip(rows, fixed_rows)
            if v118._objective_value(row, "balanced_f1")
            > v118._objective_value(baseline, "balanced_f1") + 1e-12
        )
        aggregate["folds_beating_fixed_top32_balanced"] = float(beats)
        aggregate["folds_total"] = float(len(rows))
        out[name] = aggregate
    return out


def _best_non_oracle(policies: dict[str, dict[str, float]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, metrics in policies.items():
        if name.startswith("oracle_"):
            continue
        rows.append(
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
        )
    return max(rows, key=lambda row: (row["balanced_f1"], row["pair_f1"])) if rows else {}


def _grouped_joint_eval(
    pair_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    pair_target_field: str,
    pair_epochs: int,
    pair_lr: float,
    pair_ridge_l2: float,
    max_pair_train_rows_per_fold: int,
    budget_epochs: int,
    budget_lr: float,
    seed: int,
) -> dict[str, Any]:
    del pair_epochs, pair_lr
    groups = sorted({_group_key(row) for row in candidate_rows})
    candidate_groups = {_candidate_key(row): _group_key(row) for row in candidate_rows}
    folds: list[dict[str, Any]] = []
    policy_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    selected_policy_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    pair_train_metrics: list[dict[str, float]] = []
    pair_val_metrics: list[dict[str, float]] = []
    target_rows: list[dict[str, Any]] = []

    for fold_idx, group in enumerate(groups):
        train_pairs, val_pairs = _split_pair_rows(pair_rows, candidate_groups, int(group))
        pair_model = _fit_pair_model(
            train_pairs,
            target_field=pair_target_field,
            ridge_l2=pair_ridge_l2,
            max_train_rows=max_pair_train_rows_per_fold,
            seed=seed + 13 * fold_idx,
        )
        train_pairs_pred, train_pair_metrics = _predict_pair_rows(pair_model, train_pairs, target_field=pair_target_field)
        val_pairs_pred, val_pair_metrics = _predict_pair_rows(pair_model, val_pairs, target_field=pair_target_field)
        pair_train_metrics.append(train_pair_metrics)
        pair_val_metrics.append(val_pair_metrics)

        train_candidates = [row for row in candidate_rows if _group_key(row) != group]
        val_candidates = [row for row in candidate_rows if _group_key(row) == group]
        train_records = _records_for_candidates(train_candidates, train_pairs_pred)
        val_records = _records_for_candidates(val_candidates, val_pairs_pred)
        selected_val_records = [record for record in val_records if bool(record.get("selected_by_planner", False))]
        metrics = _policy_metrics(
            train_records,
            val_records,
            epochs=budget_epochs,
            lr=budget_lr,
            seed=seed + 23 * fold_idx,
        )
        selected_metrics = _policy_metrics(
            train_records,
            selected_val_records,
            epochs=budget_epochs,
            lr=budget_lr,
            seed=seed + 23 * fold_idx,
        ) if selected_val_records else {}
        for name, row in metrics.items():
            policy_rows[name].append(row)
        for name, row in selected_metrics.items():
            selected_policy_rows[name].append(row)

        folds.append(
            {
                "group": int(group),
                "train_pair_count": int(len(train_pairs)),
                "val_pair_count": int(len(val_pairs)),
                "val_candidate_count": int(len(val_records)),
                "selected_val_candidate_count": int(len(selected_val_records)),
                "pair_train_corr": train_pair_metrics["corr"],
                "pair_val_corr": val_pair_metrics["corr"],
                "pair_val_mse": val_pair_metrics["mse"],
                "fixed_top32": metrics["fixed_top32"],
                "budget_reg_balanced_f1": metrics["budget_reg_balanced_f1"],
                "oracle_balanced_f1": metrics["oracle_balanced_f1"],
            }
        )
        for record in val_records:
            target_rows.append(
                {
                    "source_name": record["source_name"],
                    "segment_index": record["segment_index"],
                    "candidate_index": record["candidate_index"],
                    "segment_k": record["segment_k"],
                    "selected_by_planner": record["selected_by_planner"],
                    "pair_count": record["rows_count"],
                    "true_pair_count": record["true_pair_count"],
                    "best_budget": record["best_budget"],
                    "features": record["base_features"],
                    "pr_curve": {
                        str(budget): {
                            key: _as_float(value)
                            for key, value in record["curve"][budget].items()
                        }
                        for budget in v118.BUDGETS
                    },
                }
            )

    aggregated = _aggregate_policy_rows(policy_rows, fixed_rows=policy_rows["fixed_top32"])
    selected_aggregated = _aggregate_policy_rows(
        selected_policy_rows,
        fixed_rows=selected_policy_rows.get("fixed_top32", []),
    ) if selected_policy_rows else {}
    return {
        "fold_count": int(len(groups)),
        "folds": folds,
        "policies": aggregated,
        "selected_policies": selected_aggregated,
        "pair_model_train": v118._summarize(pair_train_metrics),
        "pair_model_val": v118._summarize(pair_val_metrics),
        "candidate_targets": target_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-target-field", default="calibrated_interaction_score")
    parser.add_argument("--pair-epochs", type=int, default=3)
    parser.add_argument("--pair-lr", type=float, default=0.0005)
    parser.add_argument("--pair-ridge-l2", type=float, default=1.0)
    parser.add_argument("--max-pair-train-rows-per-fold", type=int, default=40000)
    parser.add_argument("--budget-epochs", type=int, default=220)
    parser.add_argument("--budget-lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = _load_jsonl(args.pair_jsonl)
    candidate_rows = _load_jsonl(args.candidate_jsonl)
    if not pair_rows or not candidate_rows:
        raise RuntimeError("non-empty pair/candidate JSONL inputs are required")

    grouped = _grouped_joint_eval(
        pair_rows,
        candidate_rows,
        pair_target_field=args.pair_target_field,
        pair_epochs=int(args.pair_epochs),
        pair_lr=float(args.pair_lr),
        pair_ridge_l2=float(args.pair_ridge_l2),
        max_pair_train_rows_per_fold=int(args.max_pair_train_rows_per_fold),
        budget_epochs=int(args.budget_epochs),
        budget_lr=float(args.budget_lr),
        seed=int(args.seed),
    )
    fixed = grouped["policies"]["fixed_top32"]
    best = _best_non_oracle(grouped["policies"])
    summary = {
        "mode": "v119 pure-python grouped joint pair-score and PR-curve pruning selector diagnostic",
        "input_files": {
            "pair_jsonl": str(args.pair_jsonl),
            "candidate_jsonl": str(args.candidate_jsonl),
        },
        "pair_target_field": str(args.pair_target_field),
        "predicted_score_field": PRED_SCORE_FIELD,
        "pair_count": int(len(pair_rows)),
        "candidate_count": int(len(candidate_rows)),
        "group_count": int(grouped["fold_count"]),
        "pair_epochs": int(args.pair_epochs),
        "pair_solver": "closed_form_ridge",
        "pair_ridge_l2": float(args.pair_ridge_l2),
        "max_pair_train_rows_per_fold": int(args.max_pair_train_rows_per_fold),
        "budget_epochs": int(args.budget_epochs),
        "pair_model_val": grouped["pair_model_val"],
        "fixed_top32_grouped": fixed,
        "best_non_oracle_grouped": best,
        "best_non_oracle_minus_fixed_top32": {
            "pair_f1": _as_float(best.get("pair_f1", 0.0)) - _as_float(fixed.get("pair_f1", 0.0)),
            "endpoint_f1": _as_float(best.get("endpoint_f1", 0.0)) - _as_float(fixed.get("endpoint_f1", 0.0)),
            "balanced_f1": _as_float(best.get("balanced_f1", 0.0)) - _as_float(fixed.get("balanced_f1", 0.0)),
        },
        "grouped_eval": {
            key: value for key, value in grouped.items() if key != "candidate_targets"
        },
        "output_files": {
            "summary": str(args.output_dir / "stage_summary.json"),
            "candidate_targets": str(args.output_dir / "candidate_joint_targets_v119.jsonl"),
        },
    }
    (args.output_dir / "stage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _write_jsonl(args.output_dir / "candidate_joint_targets_v119.jsonl", grouped["candidate_targets"])
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
