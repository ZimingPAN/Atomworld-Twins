#!/usr/bin/env python3
"""Read-only v138 pair-score retention target diagnostic.

v135/v136 showed that budgeted projection did not cut already-hit teacher
pairs; the selected pair list almost never contained terminal vacancy pairs in
the first place.  This script consumes the full v137 factorized pair payload
and asks a narrower loader-level question: can a model-visible retention
target, trained with grouped leave-one-segment-out splits, move true terminal
vacancy pairs into a usable top-k list before we wire any new head into the
macro world model?

The diagnostic is pure Python and read-only.  It does not load torch, train a
checkpoint, or run a new long rollout.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOPK_VALUES = [8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024]
MODEL_SCORE_FIELDS = [
    "score",
    "vacancy_score",
    "energy_score",
    "source_score",
    "destination_score",
    "endpoint_sum_score",
    "interaction_residual",
    "moving_type_score",
    "order_early_score",
]
PAIR_LABELS = [
    "true_pair",
    "same_source_wrong_destination",
    "same_destination_wrong_source",
    "source_destination_unpaired",
    "false_pair",
]


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


def _position_key(value: Any) -> tuple[int, int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    return None


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _dot(weights: list[float], row: list[float]) -> float:
    return float(sum(w * x for w, x in zip(weights, row)))


def _fit_normalizer(xs: list[list[float]]) -> tuple[list[float], list[float]]:
    if not xs:
        return [], []
    dim = len(xs[0])
    means = [_mean([row[j] for row in xs]) for j in range(dim)]
    stds: list[float] = []
    for j in range(dim):
        var = _mean([(row[j] - means[j]) ** 2 for row in xs])
        stds.append(max(math.sqrt(var), 1e-6))
    return means, stds


def _normalize(row: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(row[j] - means[j]) / stds[j] for j in range(len(row))]


def _candidate_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seg_fallback, segment in enumerate(data.get("segments", [])):
        if not isinstance(segment, dict):
            continue
        segment_index = _as_int(segment.get("index", seg_fallback), seg_fallback)
        for cand_index, candidate in enumerate(segment.get("planner_candidates", [])):
            if not isinstance(candidate, dict):
                continue
            projection = candidate.get("vacancy_pair_projection_diagnostic")
            if not isinstance(projection, dict):
                continue
            pairs = projection.get("factorized_pair_scores")
            if not isinstance(pairs, list) or not pairs:
                continue
            diagnostic = candidate.get("candidate_joint_diagnostic")
            if not isinstance(diagnostic, dict):
                diagnostic = {}
            oracle = candidate.get("teacher_overlap_oracle")
            if not isinstance(oracle, dict):
                oracle = {}
            selector = candidate.get("planner_candidate_pareto_selector")
            if not isinstance(selector, dict):
                selector = {}
            selector_predictions = selector.get("predictions")
            if not isinstance(selector_predictions, dict):
                selector_predictions = {}
            record = {
                "segment_index": segment_index,
                "candidate_index": int(cand_index),
                "segment_k": _as_int(candidate.get("segment_k", diagnostic.get("segment_k", 0)), 0),
                "selected_by_planner": bool(diagnostic.get("selected_by_planner", False)),
                "selection_score": _as_float(candidate.get("selection_score", 0.0)),
                "predicted_reward_sum": _as_float(candidate.get("predicted_reward_sum", 0.0)),
                "predicted_delta_e": _as_float(candidate.get("predicted_delta_e", 0.0)),
                "predicted_expected_tau": _as_float(candidate.get("predicted_expected_tau", 0.0)),
                "predicted_noop_risk_prob": _as_float(candidate.get("predicted_noop_risk_prob", 0.0)),
                "projected_changed_count": _as_float(candidate.get("projected_changed_count", 0.0)),
                "planner_edge_completion_support_count": _as_float(
                    candidate.get("planner_edge_completion_support_count", 0.0)
                ),
                "proposal_support_mass": _as_float(candidate.get("proposal_support_mass", 0.0)),
                "proposal_support_density": _as_float(candidate.get("proposal_support_density", 0.0)),
                "candidate_quality_score": _as_float(candidate.get("candidate_quality_score", 0.0)),
                "teacher_reward_sum": _as_float(diagnostic.get("teacher_reward_sum", 0.0)),
                "site_f1": _as_float(diagnostic.get("site_f1", oracle.get("f1", 0.0))),
                "vacancy_pair_precision": _as_float(diagnostic.get("vacancy_pair_precision", 0.0)),
                "vacancy_pair_recall": _as_float(diagnostic.get("vacancy_pair_recall", 0.0)),
                "vacancy_pair_f1": _as_float(diagnostic.get("vacancy_pair_f1", 0.0)),
                "selector_budget": _as_int(selector.get("budget", 128), 128),
                "selector_pred_pair_recall": _as_float(selector_predictions.get("pair_recall", 0.0)),
                "selector_pred_pair_precision": _as_float(selector_predictions.get("pair_precision", 0.0)),
                "selector_pred_pair_f1": _as_float(selector_predictions.get("pair_f1", 0.0)),
                "pairs": [item for item in pairs if isinstance(item, dict)],
            }
            records.append(record)
    return records


def _pair_feature(record: dict[str, Any], item: dict[str, Any]) -> list[float]:
    score = _as_float(item.get("score", 0.0))
    vacancy = _as_float(item.get("vacancy_score", 0.0))
    energy = _as_float(item.get("energy_score", 0.0))
    source = _as_float(item.get("source_score", 0.0))
    destination = _as_float(item.get("destination_score", 0.0))
    endpoint = _as_float(item.get("endpoint_sum_score", 0.0))
    residual = _as_float(item.get("interaction_residual", 0.0))
    moving = _as_float(item.get("moving_type_score", 0.0))
    order = _as_float(item.get("order_early_score", 0.0))
    rank = _as_float(item.get("rank", 0.0))
    pair_count = max(len(record.get("pairs", [])), 1)
    moving_type = _as_int(item.get("moving_type", -1), -1)
    return [
        score,
        vacancy,
        energy,
        source,
        destination,
        endpoint,
        residual,
        moving,
        order,
        score - energy,
        vacancy - energy,
        source - destination,
        source * destination,
        endpoint + residual,
        moving * score,
        order * score,
        rank / float(pair_count),
        1.0 if moving_type == 0 else 0.0,
        1.0 if moving_type == 1 else 0.0,
        1.0 if moving_type == 2 else 0.0,
        math.log1p(max(_as_float(record.get("segment_k", 0.0)), 0.0)),
        _as_float(record.get("selection_score", 0.0)),
        _as_float(record.get("predicted_reward_sum", 0.0)),
        _as_float(record.get("predicted_delta_e", 0.0)),
        math.log1p(max(_as_float(record.get("predicted_expected_tau", 0.0)), 0.0)),
        _as_float(record.get("predicted_noop_risk_prob", 0.0)),
        _as_float(record.get("projected_changed_count", 0.0)) / 512.0,
        _as_float(record.get("planner_edge_completion_support_count", 0.0)) / 512.0,
        _as_float(record.get("proposal_support_mass", 0.0)),
        _as_float(record.get("proposal_support_density", 0.0)),
        _as_float(record.get("candidate_quality_score", 0.0)),
    ]


def _label(item: dict[str, Any]) -> int:
    return 1 if bool(item.get("is_true_pair", False)) else 0


def _pair_label(item: dict[str, Any]) -> str:
    if bool(item.get("is_true_pair", False)):
        return "true_pair"
    return str(item.get("pair_label", "false_pair"))


def _sample_training_rows(
    records: list[dict[str, Any]],
    *,
    negatives_per_positive_per_label: int,
    hard_top_per_positive: int,
    seed: int,
) -> list[tuple[list[float], int]]:
    rng = random.Random(seed)
    rows: list[tuple[list[float], int]] = []
    for record in records:
        pairs = [item for item in record.get("pairs", []) if isinstance(item, dict)]
        positives = [item for item in pairs if _label(item) == 1]
        if not positives:
            continue
        for item in positives:
            rows.append((_pair_feature(record, item), 1))
        negative_items: list[dict[str, Any]] = []
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in pairs:
            if _label(item) == 0:
                by_label[_pair_label(item)].append(item)
        for label_name in PAIR_LABELS:
            if label_name == "true_pair":
                continue
            items = by_label.get(label_name, [])
            if not items:
                continue
            rng.shuffle(items)
            take = min(len(items), max(1, negatives_per_positive_per_label * len(positives)))
            negative_items.extend(items[:take])
        top_false = [
            item
            for item in sorted(pairs, key=lambda row: _as_float(row.get("score", 0.0)), reverse=True)
            if _label(item) == 0
        ]
        negative_items.extend(top_false[: max(1, hard_top_per_positive * len(positives))])
        seen: set[tuple[tuple[int, int, int] | None, tuple[int, int, int] | None, int]] = set()
        for item in negative_items:
            key = (
                _position_key(item.get("source_position")),
                _position_key(item.get("destination_position")),
                _as_int(item.get("moving_type", -1), -1),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append((_pair_feature(record, item), 0))
    rng.shuffle(rows)
    return rows


def _train_logistic(
    rows: list[tuple[list[float], int]],
    *,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        return {"weights": [], "mean": [], "std": []}
    rng = random.Random(seed)
    xs = [row[0] for row in rows]
    means, stds = _fit_normalizer(xs)
    normalized = [(_normalize(x, means, stds) + [1.0], y) for x, y in rows]
    weights = [0.0 for _ in range(len(normalized[0][0]))]
    indices = list(range(len(normalized)))
    pos_count = sum(y for _, y in normalized)
    neg_count = max(len(normalized) - pos_count, 1)
    pos_weight = min(float(neg_count / max(pos_count, 1)), 16.0)
    for _epoch in range(max(int(epochs), 1)):
        rng.shuffle(indices)
        for idx in indices:
            x, y = normalized[idx]
            pred = _sigmoid(_dot(weights, x))
            weight = pos_weight if y == 1 else 1.0
            grad_scale = weight * (pred - float(y))
            for j, value in enumerate(x):
                weights[j] -= lr * (grad_scale * value + l2 * weights[j])
    return {
        "weights": weights,
        "mean": means,
        "std": stds,
        "feature_dim": len(means),
        "train_row_count": len(rows),
        "train_positive_count": int(pos_count),
        "train_negative_count": int(neg_count),
        "pos_weight": pos_weight,
    }


def _predict(model: dict[str, Any], features: list[float]) -> float:
    weights = model.get("weights", [])
    means = model.get("mean", [])
    stds = model.get("std", [])
    if not weights or not means or not stds:
        return 0.0
    row = _normalize(features, means, stds) + [1.0]
    return _sigmoid(_dot(weights, row))


def _score_records_grouped(
    records: list[dict[str, Any]],
    *,
    folds: int,
    negatives_per_positive_per_label: int,
    hard_top_per_positive: int,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> dict[tuple[int, int], list[float]]:
    fold_ids = sorted({int(record["segment_index"]) % max(int(folds), 1) for record in records})
    predictions: dict[tuple[int, int], list[float]] = {}
    for fold_id in fold_ids:
        train_records = [record for record in records if int(record["segment_index"]) % max(int(folds), 1) != fold_id]
        val_records = [record for record in records if int(record["segment_index"]) % max(int(folds), 1) == fold_id]
        train_rows = _sample_training_rows(
            train_records,
            negatives_per_positive_per_label=negatives_per_positive_per_label,
            hard_top_per_positive=hard_top_per_positive,
            seed=seed + fold_id * 7919,
        )
        model = _train_logistic(train_rows, epochs=epochs, lr=lr, l2=l2, seed=seed + fold_id * 17)
        for record in val_records:
            scores = [_predict(model, _pair_feature(record, item)) for item in record.get("pairs", [])]
            predictions[(int(record["segment_index"]), int(record["candidate_index"]))] = scores
    return predictions


def _support_metrics(selected: list[dict[str, Any]], true_items: list[dict[str, Any]]) -> dict[str, float]:
    selected_pairs = {
        (_position_key(item.get("source_position")), _position_key(item.get("destination_position")))
        for item in selected
    }
    true_pairs = {
        (_position_key(item.get("source_position")), _position_key(item.get("destination_position")))
        for item in true_items
    }
    selected_pairs.discard((None, None))
    true_pairs.discard((None, None))
    pair_tp = len(selected_pairs & true_pairs)
    pair_precision = float(pair_tp / max(len(selected_pairs), 1))
    pair_recall = float(pair_tp / max(len(true_pairs), 1))
    selected_sites: set[tuple[int, int, int]] = set()
    for item in selected:
        source = _position_key(item.get("source_position"))
        dest = _position_key(item.get("destination_position"))
        if source is not None:
            selected_sites.add(source)
        if dest is not None:
            selected_sites.add(dest)
    true_sites: set[tuple[int, int, int]] = set()
    for item in true_items:
        source = _position_key(item.get("source_position"))
        dest = _position_key(item.get("destination_position"))
        if source is not None:
            true_sites.add(source)
        if dest is not None:
            true_sites.add(dest)
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


def _rank_metrics(record: dict[str, Any], scores: list[float], *, budget: int) -> dict[str, float]:
    pairs = [item for item in record.get("pairs", []) if isinstance(item, dict)]
    if not pairs or len(scores) != len(pairs):
        return {}
    ranked = [item for _, item in sorted(zip(scores, pairs), key=lambda pair: pair[0], reverse=True)]
    true_items = [item for item in pairs if _label(item) == 1]
    selected = ranked[: max(1, min(int(budget), len(ranked)))]
    metrics = _support_metrics(selected, true_items)
    true_ranks: list[int] = []
    for rank, item in enumerate(ranked, start=1):
        if _label(item) == 1:
            true_ranks.append(rank)
    metrics.update(
        {
            "candidate_count": 1.0,
            "true_pair_found": 1.0 if true_ranks else 0.0,
            "true_pair_rank_mean": _mean([float(rank) for rank in true_ranks]),
            "true_pair_rank_percentile_mean": _mean([float(rank) / max(len(ranked), 1) for rank in true_ranks]),
            "true_pair_mrr": _mean([1.0 / float(rank) for rank in true_ranks]),
        }
    )
    for topk in TOPK_VALUES:
        top = ranked[: min(int(topk), len(ranked))]
        true_count = sum(1 for item in top if _label(item) == 1)
        metrics[f"recall_at_{topk}"] = float(true_count / max(len(true_items), 1))
        metrics[f"top{topk}_false_positive_rate"] = float((len(top) - true_count) / max(len(top), 1))
        metrics[f"top{topk}_true_pair_count"] = float(true_count)
    return metrics


def _summarize(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in metric_rows for key in row.keys()})
    return {key: _mean([_as_float(row.get(key, 0.0)) for row in metric_rows]) for key in keys}


def _evaluate_group(
    records: list[dict[str, Any]],
    learned_scores: dict[tuple[int, int], list[float]],
    *,
    budget: int,
) -> dict[str, Any]:
    base_rows: list[dict[str, float]] = []
    vacancy_rows: list[dict[str, float]] = []
    learned_rows: list[dict[str, float]] = []
    selector_budget_rows: list[dict[str, float]] = []
    for record in records:
        pairs = [item for item in record.get("pairs", []) if isinstance(item, dict)]
        if not pairs:
            continue
        key = (int(record["segment_index"]), int(record["candidate_index"]))
        base_scores = [_as_float(item.get("score", 0.0)) for item in pairs]
        vacancy_scores = [_as_float(item.get("vacancy_score", 0.0)) for item in pairs]
        predicted = learned_scores.get(key)
        if predicted is None or len(predicted) != len(pairs):
            continue
        base_rows.append(_rank_metrics(record, base_scores, budget=budget))
        vacancy_rows.append(_rank_metrics(record, vacancy_scores, budget=budget))
        learned_rows.append(_rank_metrics(record, predicted, budget=budget))
        selector_budget_rows.append(_rank_metrics(record, predicted, budget=_as_int(record.get("selector_budget", budget), budget)))
    return {
        "candidate_count": int(len(learned_rows)),
        "fixed_budget": int(budget),
        "base_score": _summarize(base_rows),
        "vacancy_score": _summarize(vacancy_rows),
        "learned_retention_score": _summarize(learned_rows),
        "learned_retention_selector_budget": _summarize(selector_budget_rows),
    }


def _candidate_rows(
    records: list[dict[str, Any]],
    learned_scores: dict[tuple[int, int], list[float]],
    *,
    budget: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        pairs = [item for item in record.get("pairs", []) if isinstance(item, dict)]
        key = (int(record["segment_index"]), int(record["candidate_index"]))
        predicted = learned_scores.get(key)
        if not pairs or predicted is None or len(predicted) != len(pairs):
            continue
        base = _rank_metrics(record, [_as_float(item.get("score", 0.0)) for item in pairs], budget=budget)
        learned = _rank_metrics(record, predicted, budget=budget)
        rows.append(
            {
                "segment_index": int(record["segment_index"]),
                "candidate_index": int(record["candidate_index"]),
                "segment_k": int(record.get("segment_k", 0)),
                "selected_by_planner": bool(record.get("selected_by_planner", False)),
                "teacher_reward_sum": _as_float(record.get("teacher_reward_sum", 0.0)),
                "site_f1": _as_float(record.get("site_f1", 0.0)),
                "selector_budget": _as_int(record.get("selector_budget", 128), 128),
                "base_recall_at_128": _as_float(base.get("recall_at_128", 0.0)),
                "base_pair_f1_top_budget": _as_float(base.get("pair_f1", 0.0)),
                "learned_recall_at_128": _as_float(learned.get("recall_at_128", 0.0)),
                "learned_pair_precision_top_budget": _as_float(learned.get("pair_precision", 0.0)),
                "learned_pair_recall_top_budget": _as_float(learned.get("pair_recall", 0.0)),
                "learned_pair_f1_top_budget": _as_float(learned.get("pair_f1", 0.0)),
                "learned_endpoint_f1_top_budget": _as_float(learned.get("endpoint_f1", 0.0)),
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _label_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for item in record.get("pairs", []):
            if isinstance(item, dict):
                counts[_pair_label(item)] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--negatives-per-positive-per-label", type=int, default=8)
    parser.add_argument("--hard-top-per-positive", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.eval_json.read_text(encoding="utf-8"))
    records = _candidate_records(data)
    if not records:
        raise RuntimeError("no factorized vacancy-pair candidate records found")
    selected = [record for record in records if bool(record.get("selected_by_planner", False))]
    learned_scores = _score_records_grouped(
        records,
        folds=max(int(args.folds), 1),
        negatives_per_positive_per_label=max(int(args.negatives_per_positive_per_label), 1),
        hard_top_per_positive=max(int(args.hard_top_per_positive), 1),
        epochs=max(int(args.epochs), 1),
        lr=float(args.lr),
        l2=float(args.l2),
        seed=int(args.seed),
    )
    summary = {
        "stage": "v138_pair_score_retention_target_readonly",
        "mode": "pure_python_grouped_loo_pair_retention_target_diagnostic",
        "eval_file": str(args.eval_json),
        "completed_rollout_segments": data.get("completed_rollout_segments"),
        "requested_rollout_segments": data.get("requested_rollout_segments"),
        "stop_reason": data.get("stop_reason"),
        "chosen_k_histogram": data.get("chosen_k_histogram"),
        "cumulative": data.get("cumulative"),
        "tau_expected": data.get("tau_expected"),
        "candidate_count": int(len(records)),
        "selected_candidate_count": int(len(selected)),
        "pair_count": int(sum(len(record.get("pairs", [])) for record in records)),
        "true_pair_count": int(sum(1 for record in records for item in record.get("pairs", []) if _label(item) == 1)),
        "label_distribution": _label_distribution(records),
        "settings": {
            "folds": int(args.folds),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "l2": float(args.l2),
            "budget": int(args.budget),
            "negatives_per_positive_per_label": int(args.negatives_per_positive_per_label),
            "hard_top_per_positive": int(args.hard_top_per_positive),
            "seed": int(args.seed),
            "feature_fields": MODEL_SCORE_FIELDS,
            "teacher_label_fields_used_as_features": False,
        },
        "all_candidates": _evaluate_group(records, learned_scores, budget=int(args.budget)),
        "selected_by_planner": _evaluate_group(selected, learned_scores, budget=int(args.budget)),
    }
    candidate_rows = _candidate_rows(records, learned_scores, budget=int(args.budget))
    _write_jsonl(args.output_dir / "candidate_retention_records_v138.jsonl", candidate_rows)
    (args.output_dir / "v138_pair_score_retention_target_readonly.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "stage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
