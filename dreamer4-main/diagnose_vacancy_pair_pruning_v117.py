#!/usr/bin/env python3
"""Pure-Python v117 pruning/count diagnostic for vacancy-pair distillation.

v116 showed that the calibrated pair-compatibility target is easy to regress,
while the support-count/pruning target is still weak.  This script stays fully
torch-free and checks whether better loader-level pruning policies can beat a
fixed top32 pair list before the target is wired into the world model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


DEFAULT_BUDGETS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512]
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


def _split_key(row: dict[str, Any]) -> int:
    return _as_int(row.get("segment_index", row.get("fold_key", 0)))


def _pair_key(row: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(_as_int(value) for value in (row.get("source_position") or [])),
        tuple(_as_int(value) for value in (row.get("destination_position") or [])),
    )


def _pair_score(row: dict[str, Any], score_field: str) -> float:
    return _as_float(row.get(score_field, row.get("calibrated_interaction_score", 0.0)))


def _true_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _as_int(row.get("label", row.get("pair_label", 0))) == 1]


def _support_metrics(selected: list[dict[str, Any]], true_rows: list[dict[str, Any]]) -> dict[str, float]:
    selected_pairs = {_pair_key(row) for row in selected}
    true_pairs = {_pair_key(row) for row in true_rows}
    pair_tp = len(selected_pairs & true_pairs)
    pair_precision = pair_tp / max(len(selected_pairs), 1)
    pair_recall = pair_tp / max(len(true_pairs), 1)

    selected_sites: set[tuple[int, ...]] = set()
    for row in selected:
        source, dest = _pair_key(row)
        selected_sites.add(source)
        selected_sites.add(dest)
    true_sites: set[tuple[int, ...]] = set()
    for row in true_rows:
        source, dest = _pair_key(row)
        true_sites.add(source)
        true_sites.add(dest)
    site_tp = len(selected_sites & true_sites)
    endpoint_precision = site_tp / max(len(selected_sites), 1)
    endpoint_recall = site_tp / max(len(true_sites), 1)
    return {
        "selected_pair_count": float(len(selected_pairs)),
        "true_pair_count": float(len(true_pairs)),
        "pair_precision": float(pair_precision),
        "pair_recall": float(pair_recall),
        "pair_f1": _f1(pair_precision, pair_recall),
        "endpoint_precision": float(endpoint_precision),
        "endpoint_recall": float(endpoint_recall),
        "endpoint_f1": _f1(endpoint_precision, endpoint_recall),
        "balanced_f1": 0.5 * _f1(pair_precision, pair_recall) + 0.5 * _f1(endpoint_precision, endpoint_recall),
    }


def _ranked(rows: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _pair_score(row, score_field), reverse=True)


def _candidate_metrics(
    rows: list[dict[str, Any]],
    score_field: str,
    selector: Callable[[list[dict[str, Any]], list[dict[str, Any]], str], list[dict[str, Any]]],
) -> dict[str, float]:
    true = _true_rows(rows)
    selected = selector(rows, true, score_field)
    if not selected and rows:
        selected = _ranked(rows, score_field)[:1]
    return _support_metrics(selected, true)


def _summarize(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in metrics for key in row.keys()})
    return {key: _mean([_as_float(row.get(key, 0.0)) for row in metrics]) for key in keys}


def _evaluate_candidates(
    candidate_rows: list[dict[str, Any]],
    pair_rows_by_key: dict[tuple[str, int, int], list[dict[str, Any]]],
    score_field: str,
    selector: Callable[[list[dict[str, Any]], list[dict[str, Any]], str], list[dict[str, Any]]],
) -> dict[str, float]:
    metrics: list[dict[str, float]] = []
    for candidate in candidate_rows:
        rows = pair_rows_by_key.get(_candidate_key(candidate), [])
        if rows:
            item = _candidate_metrics(rows, score_field, selector)
            item["candidate_count"] = 1.0
            metrics.append(item)
    return _summarize(metrics)


def _policy_score(metrics: dict[str, float], objective: str) -> float:
    if objective == "balanced_f1":
        return 0.5 * _as_float(metrics.get("pair_f1", 0.0)) + 0.5 * _as_float(metrics.get("endpoint_f1", 0.0))
    return _as_float(metrics.get(objective, 0.0))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(max(int(round(q * (len(values) - 1))), 0), len(values) - 1)
    return float(values[idx])


def _fixed_topk(k: int) -> Callable[[list[dict[str, Any]], list[dict[str, Any]], str], list[dict[str, Any]]]:
    def selector(rows: list[dict[str, Any]], _true: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
        return _ranked(rows, score_field)[: max(1, min(int(k), len(rows)))]
    return selector


def _global_threshold(threshold: float) -> Callable[[list[dict[str, Any]], list[dict[str, Any]], str], list[dict[str, Any]]]:
    def selector(rows: list[dict[str, Any]], _true: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
        selected = [row for row in rows if _pair_score(row, score_field) >= threshold]
        return selected if selected else _ranked(rows, score_field)[:1]
    return selector


def _relative_quantile(q: float) -> Callable[[list[dict[str, Any]], list[dict[str, Any]], str], list[dict[str, Any]]]:
    def selector(rows: list[dict[str, Any]], _true: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
        scores = [_pair_score(row, score_field) for row in rows]
        threshold = _percentile(scores, q)
        selected = [row for row in rows if _pair_score(row, score_field) >= threshold]
        return selected if selected else _ranked(rows, score_field)[:1]
    return selector


def _top_margin(margin: float) -> Callable[[list[dict[str, Any]], list[dict[str, Any]], str], list[dict[str, Any]]]:
    def selector(rows: list[dict[str, Any]], _true: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
        ranked = _ranked(rows, score_field)
        if not ranked:
            return []
        top = _pair_score(ranked[0], score_field)
        selected = [row for row in ranked if top - _pair_score(row, score_field) <= margin]
        return selected if selected else ranked[:1]
    return selector


def _hybrid_threshold_cap(
    threshold: float,
    cap: int,
) -> Callable[[list[dict[str, Any]], list[dict[str, Any]], str], list[dict[str, Any]]]:
    def selector(rows: list[dict[str, Any]], _true: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
        selected = [row for row in _ranked(rows, score_field) if _pair_score(row, score_field) >= threshold]
        if not selected:
            selected = _ranked(rows, score_field)[:1]
        return selected[: max(1, min(int(cap), len(selected)))]
    return selector


def _oracle_best_fixed(
    budgets: list[int],
) -> Callable[[list[dict[str, Any]], list[dict[str, Any]], str], list[dict[str, Any]]]:
    def selector(rows: list[dict[str, Any]], true: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
        ranked = _ranked(rows, score_field)
        best: tuple[float, float, float, int, list[dict[str, Any]]] | None = None
        for budget in budgets:
            selected = ranked[: max(1, min(int(budget), len(ranked)))]
            metrics = _support_metrics(selected, true)
            key = (
                _as_float(metrics.get("pair_f1", 0.0)),
                _as_float(metrics.get("endpoint_f1", 0.0)),
                _as_float(metrics.get("pair_recall", 0.0)),
                -int(metrics.get("selected_pair_count", 0.0)),
            )
            if best is None or key > best[:4]:
                best = (*key, selected)
        return (best[-1] if best else ranked[:1])
    return selector


def _oracle_true_count() -> Callable[[list[dict[str, Any]], list[dict[str, Any]], str], list[dict[str, Any]]]:
    def selector(rows: list[dict[str, Any]], true: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
        return _ranked(rows, score_field)[: max(1, min(len(true), len(rows)))]
    return selector


def _candidate_policy_targets(
    candidate_rows: list[dict[str, Any]],
    pair_rows_by_key: dict[tuple[str, int, int], list[dict[str, Any]]],
    score_field: str,
    budgets: list[int],
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        key = _candidate_key(candidate)
        rows = pair_rows_by_key.get(key, [])
        if not rows:
            continue
        true = _true_rows(rows)
        ranked = _ranked(rows, score_field)
        fixed: dict[str, dict[str, float]] = {}
        best_budget = 0
        best_metrics: dict[str, float] | None = None
        for budget in budgets:
            selected = ranked[: max(1, min(int(budget), len(ranked)))]
            metrics = _support_metrics(selected, true)
            fixed[f"top{budget}"] = metrics
            if best_metrics is None or (
                metrics["pair_f1"],
                metrics["endpoint_f1"],
                metrics["pair_recall"],
                -metrics["selected_pair_count"],
            ) > (
                best_metrics["pair_f1"],
                best_metrics["endpoint_f1"],
                best_metrics["pair_recall"],
                -best_metrics["selected_pair_count"],
            ):
                best_metrics = metrics
                best_budget = budget
        scores = [_pair_score(row, score_field) for row in rows]
        rows_out.append(
            {
                "source_name": candidate.get("source_name"),
                "segment_index": _as_int(candidate.get("segment_index", 0)),
                "candidate_index": _as_int(candidate.get("candidate_index", 0)),
                "fold_key": _as_int(candidate.get("fold_key", 0)),
                "segment_k": _as_int(candidate.get("segment_k", 0)),
                "selected_by_planner": bool(candidate.get("selected_by_planner", False)),
                "pair_count": int(len(rows)),
                "true_pair_count": int(len(true)),
                "score_quantiles": {
                    "p50": _percentile(scores, 0.50),
                    "p75": _percentile(scores, 0.75),
                    "p90": _percentile(scores, 0.90),
                    "p95": _percentile(scores, 0.95),
                    "p99": _percentile(scores, 0.99),
                    "max": max(scores) if scores else 0.0,
                },
                "fixed_budget_metrics": fixed,
                "v117_targets": {
                    "best_budget": int(best_budget),
                    "best_budget_pair_f1": _as_float((best_metrics or {}).get("pair_f1", 0.0)),
                    "best_budget_endpoint_f1": _as_float((best_metrics or {}).get("endpoint_f1", 0.0)),
                    "best_budget_selected_pair_count": _as_float((best_metrics or {}).get("selected_pair_count", 0.0)),
                },
            }
        )
    return rows_out


def _format_policy_result(
    name: str,
    family: str,
    train: dict[str, float],
    selected_train: dict[str, float],
    val: dict[str, float],
    selected_val: dict[str, float],
) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "train": train,
        "selected_train": selected_train,
        "val": val,
        "selected_val": selected_val,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-field", default="calibrated_interaction_score")
    parser.add_argument("--split-mod", type=int, default=5)
    parser.add_argument("--val-residue", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = _load_jsonl(args.pair_jsonl)
    candidate_rows = _load_jsonl(args.candidate_jsonl)
    if not pair_rows or not candidate_rows:
        raise RuntimeError("non-empty pair and candidate JSONL inputs are required")

    pair_rows_by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        pair_rows_by_key[_candidate_key(row)].append(row)

    split_mod = max(int(args.split_mod), 1)
    val_residue = int(args.val_residue) % split_mod
    train_candidates = [row for row in candidate_rows if _split_key(row) % split_mod != val_residue]
    val_candidates = [row for row in candidate_rows if _split_key(row) % split_mod == val_residue]
    if not val_candidates:
        val_candidates = candidate_rows[::split_mod]
        val_keys = {_candidate_key(row) for row in val_candidates}
        train_candidates = [row for row in candidate_rows if _candidate_key(row) not in val_keys]
    selected_train_candidates = [row for row in train_candidates if bool(row.get("selected_by_planner", False))]
    selected_val_candidates = [row for row in val_candidates if bool(row.get("selected_by_planner", False))]

    train_scores: list[float] = []
    for candidate in train_candidates:
        for row in pair_rows_by_key.get(_candidate_key(candidate), []):
            train_scores.append(_pair_score(row, args.score_field))

    policy_defs: list[tuple[str, str, Callable[[list[dict[str, Any]], list[dict[str, Any]], str], list[dict[str, Any]]]]] = []
    for budget in DEFAULT_BUDGETS:
        policy_defs.append((f"fixed_top{budget}", "fixed_topk", _fixed_topk(budget)))
    for q in [0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999]:
        threshold = _percentile(train_scores, q)
        policy_defs.append((f"global_threshold_q{q:g}", "global_threshold", _global_threshold(threshold)))
    for q in [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]:
        policy_defs.append((f"candidate_quantile_q{q:g}", "candidate_quantile", _relative_quantile(q)))
    for margin in [0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]:
        policy_defs.append((f"top_margin_{margin:g}", "top_margin", _top_margin(margin)))
    for q in [0.75, 0.90, 0.95, 0.975, 0.99]:
        threshold = _percentile(train_scores, q)
        for cap in [4, 8, 16, 32, 64, 128]:
            policy_defs.append((f"threshold_q{q:g}_cap{cap}", "threshold_cap", _hybrid_threshold_cap(threshold, cap)))
    policy_defs.append(("oracle_true_count", "oracle", _oracle_true_count()))
    policy_defs.append(("oracle_best_fixed_budget", "oracle", _oracle_best_fixed(DEFAULT_BUDGETS)))

    policies: list[dict[str, Any]] = []
    for name, family, selector in policy_defs:
        train = _evaluate_candidates(train_candidates, pair_rows_by_key, args.score_field, selector)
        selected_train = _evaluate_candidates(selected_train_candidates, pair_rows_by_key, args.score_field, selector)
        val = _evaluate_candidates(val_candidates, pair_rows_by_key, args.score_field, selector)
        selected_val = _evaluate_candidates(selected_val_candidates, pair_rows_by_key, args.score_field, selector)
        policies.append(_format_policy_result(name, family, train, selected_train, val, selected_val))

    fixed_top32 = next(item for item in policies if item["name"] == "fixed_top32")
    non_oracle = [item for item in policies if item["family"] != "oracle"]
    selected_by_objective: dict[str, dict[str, Any]] = {}
    selected_by_selected_train_objective: dict[str, dict[str, Any]] = {}
    for objective in OBJECTIVES:
        best = max(non_oracle, key=lambda item: _policy_score(item["train"], objective))
        selected_by_objective[objective] = best
        selected_best = max(non_oracle, key=lambda item: _policy_score(item["selected_train"], objective))
        selected_by_selected_train_objective[objective] = selected_best

    candidate_targets = _candidate_policy_targets(candidate_rows, pair_rows_by_key, args.score_field, DEFAULT_BUDGETS)
    train_targets = [row for row in candidate_targets if _split_key(row) % split_mod != val_residue]
    val_targets = [row for row in candidate_targets if _split_key(row) % split_mod == val_residue]

    fixed_top32_val = fixed_top32["val"]
    threshold_or_policy_beats_top32 = {
        objective: {
            "name": selected_by_objective[objective]["name"],
            "family": selected_by_objective[objective]["family"],
            "val_pair_f1_delta": _as_float(selected_by_objective[objective]["val"].get("pair_f1", 0.0))
            - _as_float(fixed_top32_val.get("pair_f1", 0.0)),
            "val_endpoint_f1_delta": _as_float(selected_by_objective[objective]["val"].get("endpoint_f1", 0.0))
            - _as_float(fixed_top32_val.get("endpoint_f1", 0.0)),
            "val_balanced_f1_delta": _policy_score(selected_by_objective[objective]["val"], "balanced_f1")
            - _policy_score(fixed_top32_val, "balanced_f1"),
        }
        for objective in OBJECTIVES
    }

    summary = {
        "mode": "v117 pure-python pruning/count calibration diagnostic",
        "score_field": args.score_field,
        "input_files": {
            "pair_jsonl": str(args.pair_jsonl),
            "candidate_jsonl": str(args.candidate_jsonl),
        },
        "pair_count": int(len(pair_rows)),
        "candidate_count": int(len(candidate_rows)),
        "train_candidate_count": int(len(train_candidates)),
        "val_candidate_count": int(len(val_candidates)),
        "selected_train_candidate_count": int(len(selected_train_candidates)),
        "selected_val_candidate_count": int(len(selected_val_candidates)),
        "fixed_top32": fixed_top32,
        "selected_by_train_objective": selected_by_objective,
        "selected_by_selected_train_objective": selected_by_selected_train_objective,
        "threshold_or_policy_beats_top32": threshold_or_policy_beats_top32,
        "oracle_reference": {
            item["name"]: item for item in policies if item["family"] == "oracle"
        },
        "top_policy_table": {
            objective: sorted(
                [
                    {
                        "name": item["name"],
                        "family": item["family"],
                        "train_objective": _policy_score(item["train"], objective),
                        "val_pair_f1": _as_float(item["val"].get("pair_f1", 0.0)),
                        "val_endpoint_f1": _as_float(item["val"].get("endpoint_f1", 0.0)),
                        "val_balanced_f1": _policy_score(item["val"], "balanced_f1"),
                        "val_selected_pair_count": _as_float(item["val"].get("selected_pair_count", 0.0)),
                    }
                    for item in non_oracle
                ],
                key=lambda row: row["train_objective"],
                reverse=True,
            )[:12]
            for objective in OBJECTIVES
        },
        "target_budget_histogram": {
            str(budget): sum(1 for row in candidate_targets if row["v117_targets"]["best_budget"] == budget)
            for budget in DEFAULT_BUDGETS
        },
        "train_target_budget_histogram": {
            str(budget): sum(1 for row in train_targets if row["v117_targets"]["best_budget"] == budget)
            for budget in DEFAULT_BUDGETS
        },
        "val_target_budget_histogram": {
            str(budget): sum(1 for row in val_targets if row["v117_targets"]["best_budget"] == budget)
            for budget in DEFAULT_BUDGETS
        },
        "output_files": {
            "summary": str(args.output_dir / "stage_summary.json"),
            "candidate_targets": str(args.output_dir / "candidate_pruning_targets_v117.jsonl"),
            "policy_table": str(args.output_dir / "policy_table_v117.json"),
        },
    }
    (args.output_dir / "stage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "policy_table_v117.json").write_text(
        json.dumps(policies, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _write_jsonl(args.output_dir / "candidate_pruning_targets_v117.jsonl", candidate_targets)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
