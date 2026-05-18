#!/usr/bin/env python3
"""Pure-Python v116 smoke for calibrated vacancy-pair distillation targets.

The remote torch runtime can be unstable during diagnostic heartbeats.  This
script deliberately avoids torch and trains small linear/softmax models over
the v115 JSONL artifacts so we can validate the pair-compatibility and
support-count targets before wiring them into the macro world model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


TOPK_BUDGETS = [4, 8, 16, 32, 64, 96, 128, 192, 256, 512]


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


def _candidate_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("source_name", "")),
        _as_int(row.get("segment_index", 0)),
        _as_int(row.get("candidate_index", 0)),
    )


def _split_key(row: dict[str, Any]) -> int:
    return _as_int(row.get("segment_index", row.get("fold_key", 0)))


def _load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
    return rows


def _pair_features(row: dict[str, Any]) -> list[float]:
    features = row.get("features")
    if isinstance(features, list) and features:
        return [_as_float(value) for value in features]
    return [
        _as_float(row.get("score", 0.0)),
        _as_float(row.get("vacancy_score", 0.0)),
        _as_float(row.get("energy_score", 0.0)),
        _as_float(row.get("source_score", 0.0)),
        _as_float(row.get("destination_score", 0.0)),
        _as_float(row.get("endpoint_sum_score", 0.0)),
        _as_float(row.get("interaction_residual", 0.0)),
        _as_float(row.get("moving_type_score", 0.0)),
        _as_float(row.get("order_early_score", 0.0)),
    ]


def _candidate_features(row: dict[str, Any]) -> list[float]:
    quantiles = row.get("calibrated_score_quantiles") or {}
    return [
        float(_as_int(row.get("pair_count", 0))),
        float(_as_int(row.get("segment_k", 0))),
        1.0 if bool(row.get("selected_by_planner", False)) else 0.0,
        _as_float(quantiles.get("p50", 0.0)),
        _as_float(quantiles.get("p75", 0.0)),
        _as_float(quantiles.get("p90", 0.0)),
        _as_float(quantiles.get("p95", 0.0)),
        _as_float(quantiles.get("p99", 0.0)),
    ]


def _candidate_features_from_scores(row: dict[str, Any], scores: list[float]) -> list[float]:
    values = sorted(scores)
    def pick(q: float) -> float:
        if not values:
            return 0.0
        idx = min(max(int(round(q * (len(values) - 1))), 0), len(values) - 1)
        return float(values[idx])
    return [
        float(_as_int(row.get("pair_count", len(scores)))),
        float(_as_int(row.get("segment_k", 0))),
        1.0 if bool(row.get("selected_by_planner", False)) else 0.0,
        pick(0.50),
        pick(0.75),
        pick(0.90),
        pick(0.95),
        pick(0.99),
    ]


def _budget_index(budget: int) -> int:
    if budget in TOPK_BUDGETS:
        return TOPK_BUDGETS.index(budget)
    return min(range(len(TOPK_BUDGETS)), key=lambda idx: abs(TOPK_BUDGETS[idx] - budget))


def _fit_normalizer(xs: list[list[float]]) -> tuple[list[float], list[float]]:
    if not xs:
        return [], []
    dim = len(xs[0])
    mean = [_mean([row[j] for row in xs]) for j in range(dim)]
    std: list[float] = []
    for j in range(dim):
        var = _mean([(row[j] - mean[j]) ** 2 for row in xs])
        std.append(max(math.sqrt(var), 1e-6))
    return mean, std


def _normalize(row: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [(row[j] - mean[j]) / std[j] for j in range(len(row))]


def _dot(weights: list[float], row: list[float]) -> float:
    return sum(w * x for w, x in zip(weights, row))


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))


def _train_linear_regressor(
    xs: list[list[float]],
    ys: list[float],
    *,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[list[float], dict[str, Any]]:
    rng = random.Random(seed)
    mean, std = _fit_normalizer(xs)
    xsn = [_normalize(row, mean, std) + [1.0] for row in xs]
    y_mean = _mean(ys)
    y_std = max(math.sqrt(_mean([(y - y_mean) ** 2 for y in ys])), 1e-6)
    ysn = [(y - y_mean) / y_std for y in ys]
    weights = [0.0 for _ in range(len(xsn[0]))]
    indices = list(range(len(xsn)))
    for _epoch in range(max(int(epochs), 1)):
        rng.shuffle(indices)
        for idx in indices:
            pred = _dot(weights, xsn[idx])
            err = pred - ysn[idx]
            for j, value in enumerate(xsn[idx]):
                weights[j] -= lr * err * value
    model = {
        "weights": weights,
        "mean": mean,
        "std": std,
        "target_mean": y_mean,
        "target_std": y_std,
    }
    return weights, model


def _predict_linear(model: dict[str, Any], xs: list[list[float]]) -> list[float]:
    mean = model["mean"]
    std = model["std"]
    weights = model["weights"]
    y_mean = float(model["target_mean"])
    y_std = float(model["target_std"])
    out: list[float] = []
    for row in xs:
        x = _normalize(row, mean, std) + [1.0]
        out.append(_dot(weights, x) * y_std + y_mean)
    return out


def _train_softmax_classifier(
    xs: list[list[float]],
    labels: list[int],
    *,
    epochs: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed + 17)
    mean, std = _fit_normalizer(xs)
    xsn = [_normalize(row, mean, std) + [1.0] for row in xs]
    classes = len(TOPK_BUDGETS)
    weights = [[0.0 for _ in range(len(xsn[0]))] for _ in range(classes)]
    indices = list(range(len(xsn)))
    for _epoch in range(max(int(epochs), 1)):
        rng.shuffle(indices)
        for idx in indices:
            logits = [_dot(w, xsn[idx]) for w in weights]
            m = max(logits)
            exp_values = [math.exp(v - m) for v in logits]
            denom = max(sum(exp_values), 1e-12)
            probs = [v / denom for v in exp_values]
            for c in range(classes):
                grad = probs[c] - (1.0 if c == labels[idx] else 0.0)
                for j, value in enumerate(xsn[idx]):
                    weights[c][j] -= lr * grad * value
    return {"weights": weights, "mean": mean, "std": std}


def _predict_softmax(model: dict[str, Any], xs: list[list[float]]) -> list[int]:
    out: list[int] = []
    for row in xs:
        x = _normalize(row, model["mean"], model["std"]) + [1.0]
        logits = [_dot(w, x) for w in model["weights"]]
        out.append(max(range(len(logits)), key=lambda idx: logits[idx]))
    return out


def _support_metrics(selected: list[dict[str, Any]], true_rows: list[dict[str, Any]]) -> dict[str, float]:
    selected_pairs = {
        (tuple(row.get("source_position") or []), tuple(row.get("destination_position") or []))
        for row in selected
    }
    true_pairs = {
        (tuple(row.get("source_position") or []), tuple(row.get("destination_position") or []))
        for row in true_rows
    }
    pair_tp = len(selected_pairs & true_pairs)
    pair_precision = float(pair_tp / max(len(selected_pairs), 1))
    pair_recall = float(pair_tp / max(len(true_pairs), 1))
    selected_sites = set()
    for row in selected:
        selected_sites.add(tuple(row.get("source_position") or []))
        selected_sites.add(tuple(row.get("destination_position") or []))
    true_sites = set()
    for row in true_rows:
        true_sites.add(tuple(row.get("source_position") or []))
        true_sites.add(tuple(row.get("destination_position") or []))
    site_tp = len(selected_sites & true_sites)
    site_precision = float(site_tp / max(len(selected_sites), 1))
    site_recall = float(site_tp / max(len(true_sites), 1))
    return {
        "selected_pair_count": float(len(selected_pairs)),
        "true_pair_count": float(len(true_pairs)),
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pair_f1": _f1(pair_precision, pair_recall),
        "endpoint_precision": site_precision,
        "endpoint_recall": site_recall,
        "endpoint_f1": _f1(site_precision, site_recall),
    }


def _summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row.keys()})
    return {key: _mean([_as_float(row.get(key, 0.0)) for row in rows]) for key in keys}


def _evaluate_policy(
    pair_rows_by_key: dict[tuple[str, int, int], list[dict[str, Any]]],
    candidate_rows: list[dict[str, Any]],
    scores_by_key: dict[tuple[str, int, int], list[float]],
    budget_by_key: dict[tuple[str, int, int], int] | None = None,
    *,
    fixed_budget: int | None = None,
) -> dict[str, float]:
    metrics: list[dict[str, float]] = []
    for row in candidate_rows:
        key = _candidate_key(row)
        pairs = pair_rows_by_key.get(key, [])
        scores = scores_by_key.get(key, [])
        if not pairs or len(pairs) != len(scores):
            continue
        budget = int(fixed_budget if fixed_budget is not None else (budget_by_key or {}).get(key, 128))
        budget = max(1, min(budget, len(pairs)))
        ranked = sorted(zip(scores, pairs), key=lambda item: item[0], reverse=True)
        selected = [item[1] for item in ranked[:budget]]
        true_rows = [item for item in pairs if _as_int(item.get("label", 0)) == 1]
        item = _support_metrics(selected, true_rows)
        item["candidate_count"] = 1.0
        metrics.append(item)
    return _summarize(metrics)


def _mse(pred: list[float], target: list[float]) -> float:
    return _mean([(p - t) ** 2 for p, t in zip(pred, target)])


def _budget_metrics(pred_idx: list[int], target_idx: list[int]) -> dict[str, float]:
    pred_budget = [TOPK_BUDGETS[idx] for idx in pred_idx]
    target_budget = [TOPK_BUDGETS[idx] for idx in target_idx]
    return {
        "acc": _mean([1.0 if p == t else 0.0 for p, t in zip(pred_idx, target_idx)]),
        "budget_mae": _mean([abs(p - t) for p, t in zip(pred_budget, target_budget)]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--count-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8192)  # accepted for script compatibility
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--pair-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = _load_jsonl(args.pair_jsonl, limit=int(args.pair_limit))
    candidate_rows = _load_jsonl(args.candidate_jsonl)
    if not pair_rows or not candidate_rows:
        raise RuntimeError("non-empty v115 JSONL inputs are required")

    train_pair_rows = [row for row in pair_rows if _split_key(row) % 5 != 0]
    val_pair_rows = [row for row in pair_rows if _split_key(row) % 5 == 0]
    train_candidate_rows = [row for row in candidate_rows if _split_key(row) % 5 != 0]
    val_candidate_rows = [row for row in candidate_rows if _split_key(row) % 5 == 0]
    if not val_pair_rows or not val_candidate_rows:
        val_pair_rows = pair_rows[::5]
        train_pair_rows = [row for idx, row in enumerate(pair_rows) if idx % 5 != 0]
        val_candidate_rows = candidate_rows[::5]
        train_candidate_rows = [row for idx, row in enumerate(candidate_rows) if idx % 5 != 0]

    train_x = [_pair_features(row) for row in train_pair_rows]
    val_x = [_pair_features(row) for row in val_pair_rows]
    train_y = [_as_float(row.get("calibrated_interaction_score", 0.0)) for row in train_pair_rows]
    val_y = [_as_float(row.get("calibrated_interaction_score", 0.0)) for row in val_pair_rows]
    _, pair_model = _train_linear_regressor(
        train_x,
        train_y,
        epochs=int(args.epochs),
        lr=float(args.lr),
        seed=int(args.seed),
    )
    train_pred = _predict_linear(pair_model, train_x)
    val_pred = _predict_linear(pair_model, val_x)

    pair_rows_by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        pair_rows_by_key[_candidate_key(row)].append(row)

    scores_by_key: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row, score in zip(train_pair_rows, train_pred):
        scores_by_key[_candidate_key(row)].append(score)
    for row, score in zip(val_pair_rows, val_pred):
        scores_by_key[_candidate_key(row)].append(score)

    count_train_x = [
        _candidate_features_from_scores(row, scores_by_key.get(_candidate_key(row), []))
        for row in train_candidate_rows
    ]
    count_val_x = [
        _candidate_features_from_scores(row, scores_by_key.get(_candidate_key(row), []))
        for row in val_candidate_rows
    ]
    count_train_y = [
        _budget_index(_as_int((row.get("support_count_targets") or {}).get("best_fixed_budget", 128), 128))
        for row in train_candidate_rows
    ]
    count_val_y = [
        _budget_index(_as_int((row.get("support_count_targets") or {}).get("best_fixed_budget", 128), 128))
        for row in val_candidate_rows
    ]
    count_model = _train_softmax_classifier(
        count_train_x,
        count_train_y,
        epochs=int(args.count_epochs),
        lr=float(args.lr),
        seed=int(args.seed),
    )
    train_count_pred = _predict_softmax(count_model, count_train_x)
    val_count_pred = _predict_softmax(count_model, count_val_x)

    budget_by_key: dict[tuple[str, int, int], int] = {}
    for row in candidate_rows:
        key = _candidate_key(row)
        scores = scores_by_key.get(key, [])
        features = _candidate_features_from_scores(row, scores)
        pred_idx = _predict_softmax(count_model, [features])[0]
        budget_by_key[key] = TOPK_BUDGETS[pred_idx]

    train_keys = {_candidate_key(row) for row in train_candidate_rows}
    val_keys = {_candidate_key(row) for row in val_candidate_rows}
    train_pair_rows_by_key = {key: value for key, value in pair_rows_by_key.items() if key in train_keys}
    val_pair_rows_by_key = {key: value for key, value in pair_rows_by_key.items() if key in val_keys}
    train_scores_by_key = {key: value for key, value in scores_by_key.items() if key in train_keys}
    val_scores_by_key = {key: value for key, value in scores_by_key.items() if key in val_keys}
    train_budget_by_key = {key: value for key, value in budget_by_key.items() if key in train_keys}
    val_budget_by_key = {key: value for key, value in budget_by_key.items() if key in val_keys}

    target_budget_histogram = {
        str(budget): sum(
            1
            for row in candidate_rows
            if TOPK_BUDGETS[
                _budget_index(_as_int((row.get("support_count_targets") or {}).get("best_fixed_budget", 128), 128))
            ]
            == budget
        )
        for budget in TOPK_BUDGETS
    }
    predicted_budget_histogram = {
        str(budget): sum(1 for value in budget_by_key.values() if value == budget) for budget in TOPK_BUDGETS
    }
    summary = {
        "mode": "v116 pure-python pair compatibility distillation and support-count smoke",
        "pair_count": len(pair_rows),
        "candidate_count": len(candidate_rows),
        "train_pair_count": len(train_pair_rows),
        "val_pair_count": len(val_pair_rows),
        "train_candidate_count": len(train_candidate_rows),
        "val_candidate_count": len(val_candidate_rows),
        "pair_model": {
            "train_mse": _mse(train_pred, train_y),
            "val_mse": _mse(val_pred, val_y),
            "train_corr": _pearson(train_pred, train_y),
            "val_corr": _pearson(val_pred, val_y),
        },
        "count_model": {
            "train": _budget_metrics(train_count_pred, count_train_y),
            "val": _budget_metrics(val_count_pred, count_val_y),
        },
        "policies": {
            "train_top128_distilled_score": _evaluate_policy(
                train_pair_rows_by_key, train_candidate_rows, train_scores_by_key, fixed_budget=128
            ),
            "val_top128_distilled_score": _evaluate_policy(
                val_pair_rows_by_key, val_candidate_rows, val_scores_by_key, fixed_budget=128
            ),
            "train_predicted_budget_distilled_score": _evaluate_policy(
                train_pair_rows_by_key, train_candidate_rows, train_scores_by_key, train_budget_by_key
            ),
            "val_predicted_budget_distilled_score": _evaluate_policy(
                val_pair_rows_by_key, val_candidate_rows, val_scores_by_key, val_budget_by_key
            ),
            "val_top32_distilled_score": _evaluate_policy(
                val_pair_rows_by_key, val_candidate_rows, val_scores_by_key, fixed_budget=32
            ),
        },
        "target_budget_histogram": target_budget_histogram,
        "predicted_budget_histogram": predicted_budget_histogram,
        "input_files": {
            "pair_jsonl": str(args.pair_jsonl),
            "candidate_jsonl": str(args.candidate_jsonl),
        },
    }
    (args.output_dir / "stage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "distill_smoke_model.json").write_text(
        json.dumps({"pair_model": pair_model, "count_model": count_model, "topk_budgets": TOPK_BUDGETS}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
