#!/usr/bin/env python3
"""Read-only v129 recall-floor support-count selector diagnostic.

v128 showed that the v123 Pareto selector budgets are too small: they improve
site/reward in the loader artifact but prune away true terminal vacancy pairs
when applied to projection.  This pure-Python diagnostic tests a more
conservative support-count objective before any closed-loop eval:

* oracle same-candidate recall-floor budgets, to check if the chosen candidate
  can be rescued by a larger budget;
* oracle all-candidate recall-floor policies, to measure the loader upper
  bound;
* grouped leave-one-segment-out predicted pair-recall floors, to test whether
  model-visible features can choose larger budgets without teacher labels.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import diagnose_candidate_pareto_selector_v122 as v122
import diagnose_vacancy_pair_selector_v119 as v119


DEFAULT_ROOT = Path("results/natural_teacher_support32_sequence_rollout_0517")
DEFAULT_V119 = DEFAULT_ROOT / "v119_pair_joint_selector_grouped_readonly_smoke" / "candidate_joint_targets_v119.jsonl"
DEFAULT_V115 = DEFAULT_ROOT / "v115_pair_interaction_distill_readonly" / "candidate_support_count_samples_v115.jsonl"
DEFAULT_V104 = DEFAULT_ROOT / "v104_candidate_two_branch_selector_readonly_smoke20" / "candidate_two_branch_samples_v104.jsonl"
DEFAULT_V123 = DEFAULT_ROOT / "v123_candidate_pareto_replay_readonly" / "candidate_pareto_replay_choices_v123.jsonl"
DEFAULT_V128 = DEFAULT_ROOT / "v128_budget_retention_readonly" / "stage_summary.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "v129_recall_floor_selector_readonly"

PREDICTED_TARGETS = (
    "teacher_reward_sum",
    "reward_norm",
    "site_f1",
    "pair_precision",
    "pair_recall",
    "pair_f1",
    "endpoint_f1",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


def _metric(row: dict[str, Any], name: str) -> float:
    return _as_float(row.get("metrics", {}).get(name, 0.0))


def _target(row: dict[str, Any], name: str) -> float:
    if name in row.get("metrics", {}):
        return _metric(row, name)
    return _as_float(row.get("targets", {}).get(name, 0.0))


def _pred(row: dict[str, Any], name: str) -> float:
    return _as_float(row.get("predictions", {}).get(name, 0.0))


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _minmax(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return lo, lo + 1.0
    return lo, hi


def _norm(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float((value - lo) / max(hi - lo, 1e-12))


def _group_rows(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_as_int(row.get("group", 0))].append(row)
    return grouped


def _candidate_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        _as_int(row.get("group", row.get("segment_index", 0))),
        _as_int(row.get("candidate_index", 0)),
        str(row.get("source_name", "")),
    )


def _load_feature_rows(
    *,
    v119_candidate_jsonl: Path,
    v115_candidate_jsonl: Path,
    v104_candidate_jsonl: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, join_summary = v122._feature_rows(
        v119_candidate_jsonl=v119_candidate_jsonl,
        v115_candidate_jsonl=v115_candidate_jsonl,
        v104_candidate_jsonl=v104_candidate_jsonl,
    )
    if not rows:
        raise RuntimeError("no v129 candidate-budget rows")
    return rows, join_summary


def _selected_fixed(rows: list[dict[str, Any]], budget: int = 32) -> dict[str, Any]:
    selected = [
        row for row in rows if bool(row.get("selected_by_planner", False)) and _as_int(row.get("budget")) == budget
    ]
    if selected:
        return selected[0]
    matching = [row for row in rows if _as_int(row.get("budget")) == budget]
    return matching[0] if matching else rows[0]


def _score_actual_balanced(row: dict[str, Any]) -> float:
    return float(
        0.24 * _metric(row, "reward_norm")
        + 0.22 * _metric(row, "site_f1")
        + 0.18 * _metric(row, "pair_precision")
        + 0.22 * _metric(row, "pair_f1")
        + 0.14 * _metric(row, "endpoint_f1")
    )


def _score_pred_balanced(row: dict[str, Any], stats: dict[str, tuple[float, float]]) -> float:
    reward = 0.5 * _norm(_pred(row, "teacher_reward_sum"), stats["teacher_reward_sum"]) + 0.5 * _norm(
        _pred(row, "reward_norm"), stats["reward_norm"]
    )
    return float(
        0.24 * reward
        + 0.22 * _norm(_pred(row, "site_f1"), stats["site_f1"])
        + 0.18 * _norm(_pred(row, "pair_precision"), stats["pair_precision"])
        + 0.22 * _norm(_pred(row, "pair_f1"), stats["pair_f1"])
        + 0.14 * _norm(_pred(row, "endpoint_f1"), stats["endpoint_f1"])
    )


def _fit_fold_models(train_rows: list[dict[str, Any]], *, ridge_l2: float) -> dict[str, dict[str, Any]]:
    xs = [row["features"] for row in train_rows]
    models: dict[str, dict[str, Any]] = {}
    for target in PREDICTED_TARGETS:
        ys = [_target(row, target) for row in train_rows]
        models[target] = v119._fit_ridge_regressor(xs, ys, l2=ridge_l2)
    return models


def _attach_fold_predictions(rows: list[dict[str, Any]], models: dict[str, dict[str, Any]]) -> None:
    xs = [row["features"] for row in rows]
    for target, model in models.items():
        preds = v119._predict_ridge(model, xs)
        for row, pred in zip(rows, preds):
            row.setdefault("predictions", {})[target] = float(pred)


def _prediction_stats(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    return {target: _minmax([_pred(row, target) for row in rows]) for target in PREDICTED_TARGETS}


def _pick_oracle_recall_floor(rows: list[dict[str, Any]], floor: float) -> dict[str, Any]:
    eligible = [row for row in rows if _metric(row, "pair_recall") >= floor]
    pool = eligible if eligible else rows
    return max(pool, key=lambda row: (_score_actual_balanced(row), -_metric(row, "selected_pair_count")))


def _pick_pred_recall_floor(rows: list[dict[str, Any]], floor: float) -> dict[str, Any]:
    stats = _prediction_stats(rows)
    eligible = [row for row in rows if _pred(row, "pair_recall") >= floor]
    pool = eligible if eligible else rows
    return max(pool, key=lambda row: (_score_pred_balanced(row, stats), -_metric(row, "selected_pair_count")))


def _match_v123_choice(
    rows: list[dict[str, Any]],
    choice: dict[str, Any],
    *,
    fallback_budget: int | None = None,
) -> dict[str, Any]:
    group = _as_int(choice.get("group", 0))
    cand = _as_int(choice.get("candidate_index", 0))
    source = str(choice.get("source_name", ""))
    budget = _as_int(choice.get("budget", fallback_budget if fallback_budget is not None else 32))
    matches = [
        row
        for row in rows
        if _as_int(row.get("group", 0)) == group
        and _as_int(row.get("candidate_index", 0)) == cand
        and str(row.get("source_name", "")) == source
        and _as_int(row.get("budget", 0)) == budget
    ]
    if matches:
        return matches[0]
    same_candidate = [
        row
        for row in rows
        if _as_int(row.get("group", 0)) == group
        and _as_int(row.get("candidate_index", 0)) == cand
        and str(row.get("source_name", "")) == source
    ]
    if same_candidate:
        return min(same_candidate, key=lambda row: abs(_as_int(row.get("budget", 0)) - budget))
    return _selected_fixed(rows, budget=32)


def _same_candidate_min_recall_floor(
    rows: list[dict[str, Any]],
    choice: dict[str, Any],
    *,
    floor: float,
) -> dict[str, Any]:
    group = _as_int(choice.get("group", 0))
    cand = _as_int(choice.get("candidate_index", 0))
    source = str(choice.get("source_name", ""))
    same_candidate = [
        row
        for row in rows
        if _as_int(row.get("group", 0)) == group
        and _as_int(row.get("candidate_index", 0)) == cand
        and str(row.get("source_name", "")) == source
    ]
    if not same_candidate:
        return _selected_fixed(rows, budget=32)
    eligible = [row for row in same_candidate if _metric(row, "pair_recall") >= floor]
    if eligible:
        return min(eligible, key=lambda row: (_metric(row, "selected_pair_count"), -_score_actual_balanced(row)))
    return max(same_candidate, key=lambda row: (_score_actual_balanced(row), -_metric(row, "selected_pair_count")))


def _row_preview(row: dict[str, Any], policy: str) -> dict[str, Any]:
    return {
        "policy": policy,
        "group": _as_int(row.get("group", 0)),
        "candidate_index": _as_int(row.get("candidate_index", 0)),
        "source_name": str(row.get("source_name", "")),
        "segment_k": _as_int(row.get("segment_k", 0)),
        "budget": _as_int(row.get("budget", 0)),
        "teacher_reward_sum": _metric(row, "teacher_reward_sum"),
        "site_f1": _metric(row, "site_f1"),
        "pair_precision": _metric(row, "pair_precision"),
        "pair_recall": _metric(row, "pair_recall"),
        "pair_f1": _metric(row, "pair_f1"),
        "endpoint_f1": _metric(row, "endpoint_f1"),
        "selected_pair_count": _metric(row, "selected_pair_count"),
        "pred_pair_recall": _pred(row, "pair_recall") if row.get("predictions") else None,
    }


def _summarize(rows: list[dict[str, Any]], baseline: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not rows:
        return {}
    summary = {
        "count": len(rows),
        "avg_teacher_reward_sum": _mean([_metric(row, "teacher_reward_sum") for row in rows]),
        "avg_site_f1": _mean([_metric(row, "site_f1") for row in rows]),
        "avg_pair_precision": _mean([_metric(row, "pair_precision") for row in rows]),
        "avg_pair_recall": _mean([_metric(row, "pair_recall") for row in rows]),
        "avg_pair_f1": _mean([_metric(row, "pair_f1") for row in rows]),
        "avg_endpoint_f1": _mean([_metric(row, "endpoint_f1") for row in rows]),
        "avg_selected_pair_count": _mean([_metric(row, "selected_pair_count") for row in rows]),
        "budget_histogram": dict(sorted(Counter(_as_int(row.get("budget", 0)) for row in rows).items())),
        "source_histogram": dict(sorted(Counter(str(row.get("source_name", "")) for row in rows).items())),
    }
    if baseline and len(baseline) == len(rows):
        summary["minus_selected_fixed_top32"] = {
            "avg_teacher_reward_sum": summary["avg_teacher_reward_sum"]
            - _mean([_metric(row, "teacher_reward_sum") for row in baseline]),
            "avg_site_f1": summary["avg_site_f1"] - _mean([_metric(row, "site_f1") for row in baseline]),
            "avg_pair_precision": summary["avg_pair_precision"]
            - _mean([_metric(row, "pair_precision") for row in baseline]),
            "avg_pair_recall": summary["avg_pair_recall"] - _mean([_metric(row, "pair_recall") for row in baseline]),
            "avg_pair_f1": summary["avg_pair_f1"] - _mean([_metric(row, "pair_f1") for row in baseline]),
            "avg_endpoint_f1": summary["avg_endpoint_f1"] - _mean([_metric(row, "endpoint_f1") for row in baseline]),
            "avg_selected_pair_count": summary["avg_selected_pair_count"]
            - _mean([_metric(row, "selected_pair_count") for row in baseline]),
        }
        wins = 0
        for row, base in zip(rows, baseline):
            if (
                _metric(row, "teacher_reward_sum") >= _metric(base, "teacher_reward_sum") - 1e-12
                and _metric(row, "site_f1") > _metric(base, "site_f1") + 1e-12
                and _metric(row, "pair_precision") > _metric(base, "pair_precision") + 1e-12
                and _metric(row, "pair_f1") > _metric(base, "pair_f1") + 1e-12
            ):
                wins += 1
        summary["folds_improving_reward_site_pair_precision_f1"] = wins
    return summary


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))


def run(
    *,
    v119_candidate_jsonl: Path,
    v115_candidate_jsonl: Path,
    v104_candidate_jsonl: Path,
    v123_choices_jsonl: Path,
    v128_summary_json: Path | None,
    output_dir: Path,
    ridge_l2: float,
) -> dict[str, Any]:
    rows, join_summary = _load_feature_rows(
        v119_candidate_jsonl=v119_candidate_jsonl,
        v115_candidate_jsonl=v115_candidate_jsonl,
        v104_candidate_jsonl=v104_candidate_jsonl,
    )
    grouped = _group_rows(rows)
    v123_choices = [row for row in _load_jsonl(v123_choices_jsonl) if row.get("policy") == "pred_pareto_balanced"]
    v123_by_group = {_as_int(row.get("group", 0)): row for row in v123_choices}

    policies: dict[str, list[dict[str, Any]]] = defaultdict(list)
    previews: list[dict[str, Any]] = []
    val_pred_rows: list[dict[str, Any]] = []

    for group, group_rows in sorted(grouped.items()):
        train_rows = [row for row in rows if _as_int(row.get("group", 0)) != group]
        models = _fit_fold_models(train_rows, ridge_l2=ridge_l2)
        _attach_fold_predictions(group_rows, models)
        val_pred_rows.extend(group_rows)

        fixed = _selected_fixed(group_rows, budget=32)
        policies["selected_fixed_top32"].append(fixed)

        choice = v123_by_group.get(group)
        if choice is not None:
            policies["v123_pred_pareto_balanced"].append(_match_v123_choice(group_rows, choice))
            policies["v123_same_candidate_min_recall0p5"].append(
                _same_candidate_min_recall_floor(group_rows, choice, floor=0.5)
            )
            policies["v123_same_candidate_min_recall0p8"].append(
                _same_candidate_min_recall_floor(group_rows, choice, floor=0.8)
            )

        for floor in (0.5, 0.6, 0.8):
            policies[f"oracle_recall_floor{str(floor).replace('.', 'p')}_balanced"].append(
                _pick_oracle_recall_floor(group_rows, floor)
            )
        for floor in (0.3, 0.4, 0.5, 0.6):
            policies[f"loo_pred_recall_floor{str(floor).replace('.', 'p')}_balanced"].append(
                _pick_pred_recall_floor(group_rows, floor)
            )

    baseline = policies["selected_fixed_top32"]
    policy_eval = {name: _summarize(picks, baseline) for name, picks in sorted(policies.items())}
    for name, picks in sorted(policies.items()):
        for row in picks:
            previews.append(_row_preview(row, name))

    non_oracle_sync = [
        {"name": name, **summary}
        for name, summary in sorted(policy_eval.items())
        if name.startswith("loo_")
        and summary.get("minus_selected_fixed_top32", {}).get("avg_teacher_reward_sum", -1.0) >= -1e-12
        and summary.get("minus_selected_fixed_top32", {}).get("avg_site_f1", -1.0) > 1e-12
        and summary.get("minus_selected_fixed_top32", {}).get("avg_pair_precision", -1.0) > 1e-12
        and summary.get("minus_selected_fixed_top32", {}).get("avg_pair_f1", -1.0) > 1e-12
    ]

    prediction_quality = {
        target: _corr([_pred(row, target) for row in val_pred_rows], [_target(row, target) for row in val_pred_rows])
        for target in PREDICTED_TARGETS
    }
    v128_summary = json.loads(v128_summary_json.read_text(encoding="utf-8")) if v128_summary_json and v128_summary_json.exists() else {}

    output_dir.mkdir(parents=True, exist_ok=True)
    choices_path = output_dir / "candidate_recall_floor_choices_v129.jsonl"
    _write_jsonl(choices_path, previews)
    best_non_oracle = max(
        ({"name": name, **summary} for name, summary in policy_eval.items() if name.startswith("loo_")),
        key=lambda row: (
            row["avg_teacher_reward_sum"] >= policy_eval["selected_fixed_top32"]["avg_teacher_reward_sum"],
            row["avg_site_f1"]
            + row["avg_pair_precision"]
            + row["avg_pair_f1"]
            + 0.2 * row["avg_pair_recall"],
        ),
    )
    summary = {
        "mode": "v129 pure-python recall-floor support-count selector diagnostic",
        "input_files": {
            "v119_candidate_jsonl": str(v119_candidate_jsonl),
            "v115_candidate_jsonl": str(v115_candidate_jsonl),
            "v104_candidate_jsonl": str(v104_candidate_jsonl),
            "v123_choices_jsonl": str(v123_choices_jsonl),
            "v128_summary_json": str(v128_summary_json) if v128_summary_json else "",
        },
        "candidate_budget_row_count": len(rows),
        "candidate_count": len({tuple(row.get("key", ())) for row in rows}),
        "group_count": len(grouped),
        "ridge_l2": float(ridge_l2),
        "join_summary": join_summary,
        "selected_fixed_top32": policy_eval["selected_fixed_top32"],
        "best_non_oracle": best_non_oracle,
        "non_oracle_average_sync_improvements": non_oracle_sync,
        "policy_eval": policy_eval,
        "prediction_quality_corr": prediction_quality,
        "v128_focus_reference": (v128_summary.get("policy_summary") or {}).get("pred_pareto_balanced", {}),
        "output_files": {
            "choices": str(choices_path),
            "summary": str(output_dir / "stage_summary.json"),
        },
        "current_judgement": (
            "A predicted recall-floor policy is only a candidate for long-eval smoke if it "
            "beats selected_fixed_top32 on reward, site F1, pair precision, and pair F1 in "
            "grouped loader evaluation. Oracle recall floors remain upper bounds and must "
            "not be treated as deployable."
        ),
    }
    (output_dir / "stage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v119-candidate-jsonl", type=Path, default=DEFAULT_V119)
    parser.add_argument("--v115-candidate-jsonl", type=Path, default=DEFAULT_V115)
    parser.add_argument("--v104-candidate-jsonl", type=Path, default=DEFAULT_V104)
    parser.add_argument("--v123-choices-jsonl", type=Path, default=DEFAULT_V123)
    parser.add_argument("--v128-summary-json", type=Path, default=DEFAULT_V128)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    args = parser.parse_args()
    summary = run(
        v119_candidate_jsonl=args.v119_candidate_jsonl,
        v115_candidate_jsonl=args.v115_candidate_jsonl,
        v104_candidate_jsonl=args.v104_candidate_jsonl,
        v123_choices_jsonl=args.v123_choices_jsonl,
        v128_summary_json=args.v128_summary_json,
        output_dir=args.output_dir,
        ridge_l2=args.ridge_l2,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
