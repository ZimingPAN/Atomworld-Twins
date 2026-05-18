#!/usr/bin/env python3
"""Read-only v132 live calibration / feature-drift diagnostic.

v131 proved that the live recall-floor selector hook is deployable in
diagnostic mode, but the live selector predicted very high pair recall at tiny
budgets.  This pure-Python diagnostic compares the v130 offline 45-D feature
space and recall-floor replay against the v131 live long-eval candidate
payload.  It does not train a checkpoint, run torch, or change planner output.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import diagnose_candidate_recall_floor_v129 as v129


DEFAULT_ROOT = Path("results/natural_teacher_support32_sequence_rollout_0517")
DEFAULT_V119 = DEFAULT_ROOT / "v119_pair_joint_selector_grouped_readonly_smoke" / "candidate_joint_targets_v119.jsonl"
DEFAULT_V115 = DEFAULT_ROOT / "v115_pair_interaction_distill_readonly" / "candidate_support_count_samples_v115.jsonl"
DEFAULT_V104 = DEFAULT_ROOT / "v104_candidate_two_branch_selector_readonly_smoke20" / "candidate_two_branch_samples_v104.jsonl"
DEFAULT_V130_DIR = DEFAULT_ROOT / "v130_recall_floor_selector_replay_readonly"
DEFAULT_V130_SPEC = DEFAULT_V130_DIR / "candidate_recall_floor_selector_spec_v130.json"
DEFAULT_V130_SUMMARY = DEFAULT_V130_DIR / "stage_summary.json"
DEFAULT_V130_CHOICES = DEFAULT_V130_DIR / "candidate_recall_floor_replay_choices_v130.jsonl"
DEFAULT_V131_DIR = DEFAULT_ROOT / "v131_recall_floor_selector_diagnostic_smoke20"
DEFAULT_V131_EVAL = DEFAULT_V131_DIR / "eval_long_recall_floor_diagnostic_20.json"
DEFAULT_V131_SUMMARY = DEFAULT_V131_DIR / "stage_summary.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "v132_live_calibration_feature_drift_readonly"

PARETO_BUDGETS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512]
PARETO_TARGETS = (
    "teacher_reward_sum",
    "reward_norm",
    "site_f1",
    "pair_precision",
    "pair_f1",
    "endpoint_f1",
    "pair_recall",
)
FEATURE_NAMES = [
    "bias",
    "pair_count",
    "segment_k",
    "candidate_index",
    "selected_by_preselector",
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
    "score_gap_rank4",
    "score_gap_rank8",
    "score_gap_rank16",
    "score_gap_rank32",
    "score_gap_rank64",
    "score_gap_rank128",
    "log_budget",
    "budget",
    "budget_ratio",
    "sqrt_budget",
    "inverse_budget",
    "v104_bias",
    "v104_pre_score_norm",
    "v104_reward_norm",
    "v104_delta_norm",
    "v104_tau_inv_norm",
    "v104_noop_inv_norm",
    "v104_quality_norm",
    "v104_projected_inv_norm",
    "v104_pair_inv_norm",
    "v104_density_inv_norm",
    "v104_k_norm",
]


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(mean(finite)) if finite else 0.0


def _std(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0
    mu = _mean(finite)
    return float(math.sqrt(sum((value - mu) ** 2 for value in finite) / len(finite)))


def _quantile(values: list[float], q: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return 0.0
    idx = min(max(int(round(float(q) * (len(finite) - 1))), 0), len(finite) - 1)
    return float(finite[idx])


def _hist(values: list[int | float | str]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items(), key=lambda item: (float(item[0]) if item[0].replace(".", "", 1).replace("-", "", 1).isdigit() else item[0])))


def _metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": _mean(values),
        "std": _std(values),
        "min": _quantile(values, 0.0),
        "p25": _quantile(values, 0.25),
        "p50": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "max": _quantile(values, 1.0),
    }


def _minmax(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0, 1.0
    lo = min(finite)
    hi = max(finite)
    if abs(hi - lo) < 1e-12:
        return lo, lo + 1.0
    return lo, hi


def _norm(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float((float(value) - lo) / max(float(hi) - float(lo), 1e-12))


def _score_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = min(max(int(round(float(q) * (len(values) - 1))), 0), len(values) - 1)
    return float(sorted(values)[idx])


def _score_at_rank(desc_scores: list[float], rank: int) -> float:
    if not desc_scores:
        return 0.0
    idx = min(max(int(rank) - 1, 0), len(desc_scores) - 1)
    return float(desc_scores[idx])


def _predict(spec: dict[str, Any], target: str, features: list[float]) -> float:
    model = (spec.get("models") or {}).get(target)
    if not isinstance(model, dict):
        return 0.0
    mean_vec = [float(value) for value in model.get("mean", [])]
    std_vec = [max(float(value), 1e-12) for value in model.get("std", [])]
    weights = [float(value) for value in model.get("weights", [])]
    if len(features) != len(mean_vec) or len(weights) != len(mean_vec) + 1:
        return 0.0
    xs = [(float(value) - mean_vec[idx]) / std_vec[idx] for idx, value in enumerate(features)] + [1.0]
    raw = sum(weight * value for weight, value in zip(weights, xs))
    return float(raw) * _as_float(model.get("target_std", 0.0)) + _as_float(model.get("target_mean", 0.0))


def _prediction_values(row: dict[str, Any], stats: dict[str, tuple[float, float]]) -> dict[str, float]:
    preds = row.get("predictions") or {}
    reward_raw = _norm(_as_float(preds.get("teacher_reward_sum", 0.0)), stats["teacher_reward_sum"])
    reward_rel = _norm(_as_float(preds.get("reward_norm", 0.0)), stats["reward_norm"])
    return {
        "reward": 0.5 * reward_raw + 0.5 * reward_rel,
        "site": _norm(_as_float(preds.get("site_f1", 0.0)), stats["site_f1"]),
        "pair_precision": _norm(_as_float(preds.get("pair_precision", 0.0)), stats["pair_precision"]),
        "pair_f1": _norm(_as_float(preds.get("pair_f1", 0.0)), stats["pair_f1"]),
        "endpoint": _norm(_as_float(preds.get("endpoint_f1", 0.0)), stats["endpoint_f1"]),
    }


def _balanced_score(row: dict[str, Any], stats: dict[str, tuple[float, float]]) -> float:
    values = _prediction_values(row, stats)
    return float(
        0.24 * values["reward"]
        + 0.22 * values["site"]
        + 0.22 * values["pair_f1"]
        + 0.18 * values["pair_precision"]
        + 0.14 * values["endpoint"]
    )


def _attach_predictions(rows: list[dict[str, Any]], spec: dict[str, Any]) -> None:
    for row in rows:
        row["predictions"] = {
            target: _predict(spec, target, [float(value) for value in row.get("features", [])])
            for target in PARETO_TARGETS
        }


def _group_rows(rows: list[dict[str, Any]], key: str = "group") -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_as_int(row.get(key, 0))].append(row)
    return grouped


def _pick_recall_floor(rows: list[dict[str, Any]], floor: float) -> dict[str, Any]:
    stats = {target: _minmax([_as_float(row.get("predictions", {}).get(target, 0.0)) for row in rows]) for target in PARETO_TARGETS}
    for row in rows:
        row["selector_score"] = _balanced_score(row, stats)
    eligible = [row for row in rows if _as_float(row.get("predictions", {}).get("pair_recall", 0.0)) >= floor]
    pool = eligible if eligible else rows
    return max(pool, key=lambda row: (_as_float(row.get("selector_score", 0.0)), -_as_float(row.get("budget", 0.0))))


def _offline_rows(
    *,
    v119_candidate_jsonl: Path,
    v115_candidate_jsonl: Path,
    v104_candidate_jsonl: Path,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, _ = v129._load_feature_rows(
        v119_candidate_jsonl=v119_candidate_jsonl,
        v115_candidate_jsonl=v115_candidate_jsonl,
        v104_candidate_jsonl=v104_candidate_jsonl,
    )
    for row in rows:
        row["domain"] = "offline_v130"
    _attach_predictions(rows, spec)
    picks = [_pick_recall_floor(group_rows, 0.6) for _, group_rows in sorted(_group_rows(rows).items())]
    return rows, picks


def _pair_scores(candidate: dict[str, Any], pair_score_field: str = "score") -> list[float]:
    diagnostic = candidate.get("vacancy_pair_projection_diagnostic")
    if not isinstance(diagnostic, dict):
        return []
    pairs = diagnostic.get("factorized_pair_scores")
    if not isinstance(pairs, list):
        return []
    return [_as_float(item.get(pair_score_field, item.get("score", 0.0))) for item in pairs if isinstance(item, dict)]


def _candidate_family_flags(candidate: dict[str, Any], eval_payload: dict[str, Any]) -> tuple[float, float, float]:
    source = str(candidate.get("planner_projection_change_source", eval_payload.get("planner_projection_change_source", "")))
    anchor = str(candidate.get("planner_edge_completion_anchor_source", eval_payload.get("planner_edge_completion_anchor_source", "")))
    destination = str(
        candidate.get("planner_edge_completion_destination_source", eval_payload.get("planner_edge_completion_destination_source", ""))
    )
    text = " ".join([source, anchor, destination])
    diagnostic = candidate.get("vacancy_pair_projection_diagnostic")
    return (
        1.0 if "vacancy" in text else 0.0,
        1.0 if "energy" in text else 0.0,
        1.0 if isinstance(diagnostic, dict) and isinstance(diagnostic.get("factorized_pair_scores"), list) else 0.0,
    )


def _curve_features(
    candidate: dict[str, Any],
    *,
    eval_payload: dict[str, Any],
    candidate_index: int,
    selected_by_preselector: bool,
    pair_scores: list[float],
) -> list[float]:
    desc = sorted([float(value) for value in pair_scores], reverse=True)
    if not desc:
        desc = [0.0]
    score_mean = _mean(desc)
    score_std = _std(desc)
    max_score = _score_at_rank(desc, 1)
    source_is_vacancy, source_is_energy, source_is_factorized = _candidate_family_flags(candidate, eval_payload)
    return [
        1.0,
        float(len(pair_scores)),
        float(_as_int(candidate.get("segment_k", 0))),
        float(candidate_index),
        1.0 if selected_by_preselector else 0.0,
        source_is_vacancy,
        source_is_energy,
        source_is_factorized,
        max_score,
        _score_at_rank(desc, 2),
        _score_at_rank(desc, 4),
        _score_at_rank(desc, 8),
        _score_at_rank(desc, 16),
        _score_at_rank(desc, 32),
        _score_at_rank(desc, 64),
        _score_at_rank(desc, 128),
        _score_percentile(desc, 0.50),
        _score_percentile(desc, 0.75),
        _score_percentile(desc, 0.90),
        _score_percentile(desc, 0.95),
        _score_percentile(desc, 0.99),
        score_mean,
        score_std,
        max_score - _score_at_rank(desc, 4),
        max_score - _score_at_rank(desc, 8),
        max_score - _score_at_rank(desc, 16),
        max_score - _score_at_rank(desc, 32),
        max_score - _score_at_rank(desc, 64),
        max_score - _score_at_rank(desc, 128),
    ]


def _metric_features(base_features: list[float], budget: int, pair_count: int) -> list[float]:
    budget_value = max(int(budget), 1)
    return list(base_features) + [
        math.log(float(budget_value)),
        float(budget_value),
        float(budget_value / max(int(pair_count), 1)),
        math.sqrt(float(budget_value)),
        1.0 / float(budget_value),
    ]


def _planner_visible_features(candidates: list[dict[str, Any]], selected_by_preselector: dict[int, bool]) -> dict[int, list[float]]:
    pre_scores = [_as_float(candidate.get("pre_pareto_selection_score", candidate.get("selection_score", 0.0))) for candidate in candidates]
    reward_values = [_as_float(candidate.get("predicted_reward_sum", 0.0)) for candidate in candidates]
    delta_values = [_as_float(candidate.get("predicted_delta_e", 0.0)) for candidate in candidates]
    tau_inv_values = [1.0 / max(_as_float(candidate.get("predicted_expected_tau", 0.0)), 1e-12) for candidate in candidates]
    noop_inv_values = [1.0 - _as_float(candidate.get("predicted_noop_risk_prob", 0.0)) for candidate in candidates]
    quality_values = [_as_float(candidate.get("candidate_quality_score", 0.0)) for candidate in candidates]
    projected_inv_values = [1.0 / (1.0 + max(_as_float(candidate.get("projected_changed_count", 0.0)), 0.0)) for candidate in candidates]
    pair_inv_values = []
    for candidate in candidates:
        diagnostic = candidate.get("vacancy_pair_projection_diagnostic") if isinstance(candidate.get("vacancy_pair_projection_diagnostic"), dict) else {}
        pair_inv_values.append(
            1.0
            / (
                1.0
                + max(
                    _as_float(diagnostic.get("selected_pair_count", candidate.get("planner_edge_completion_support_count", 0.0))),
                    0.0,
                )
            )
        )
    density_inv_values = [1.0 - _as_float(candidate.get("proposal_support_density", 0.0)) for candidate in candidates]
    k_values = [_as_float(candidate.get("segment_k", 0.0)) for candidate in candidates]
    spans = {
        "pre": _minmax(pre_scores),
        "reward": _minmax(reward_values),
        "delta": _minmax(delta_values),
        "tau": _minmax(tau_inv_values),
        "noop": _minmax(noop_inv_values),
        "quality": _minmax(quality_values),
        "projected": _minmax(projected_inv_values),
        "pair": _minmax(pair_inv_values),
        "density": _minmax(density_inv_values),
        "k": _minmax(k_values),
    }
    features: dict[int, list[float]] = {}
    for idx, candidate in enumerate(candidates):
        features[id(candidate)] = [
            1.0,
            _norm(pre_scores[idx], spans["pre"]),
            _norm(reward_values[idx], spans["reward"]),
            _norm(delta_values[idx], spans["delta"]),
            _norm(tau_inv_values[idx], spans["tau"]),
            _norm(noop_inv_values[idx], spans["noop"]),
            _norm(quality_values[idx], spans["quality"]),
            _norm(projected_inv_values[idx], spans["projected"]),
            _norm(pair_inv_values[idx], spans["pair"]),
            _norm(density_inv_values[idx], spans["density"]),
            _norm(k_values[idx], spans["k"]),
        ]
    return features


def _legal(candidate: dict[str, Any]) -> bool:
    return _as_float(candidate.get("reachability_violation", 1.0)) <= 0.0 and _as_float(candidate.get("projected_changed_count", 0.0)) >= 2.0


def _live_rows(
    eval_payload: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    recorded_candidates: list[dict[str, Any]] = []
    for segment_idx, segment in enumerate(eval_payload.get("segments", [])):
        group = _as_int(segment.get("index", segment_idx))
        candidates = [candidate for candidate in segment.get("planner_candidates", []) if isinstance(candidate, dict)]
        legal = [candidate for candidate in candidates if _legal(candidate)]
        if not legal:
            continue
        preselected = max(legal, key=lambda item: _as_float(item.get("selection_score", -1e100)))
        selected_by_preselector = {id(candidate): bool(candidate is preselected) for candidate in legal}
        v104_features = _planner_visible_features(legal, selected_by_preselector)
        for candidate_index, candidate in enumerate(legal):
            pair_scores = _pair_scores(candidate)
            if not pair_scores:
                continue
            base = _curve_features(
                candidate,
                eval_payload=eval_payload,
                candidate_index=candidate_index,
                selected_by_preselector=selected_by_preselector.get(id(candidate), False),
                pair_scores=pair_scores,
            )
            extra = v104_features.get(id(candidate), [0.0] * 11)
            selector = candidate.get("planner_candidate_pareto_selector") if isinstance(candidate.get("planner_candidate_pareto_selector"), dict) else {}
            recorded_candidates.append(
                {
                    "domain": "live_v131_recorded_candidate",
                    "group": group,
                    "candidate_index": candidate_index,
                    "segment_k": _as_int(candidate.get("segment_k", 0)),
                    "budget": _as_int(selector.get("budget", 0)),
                    "selected_pair_count": float(min(_as_int(selector.get("budget", 0)), len(pair_scores))),
                    "selector_score": _as_float(selector.get("score", 0.0)),
                    "features": base + [0.0] * 5 + extra,
                    "predictions": selector.get("predictions", {}),
                    "pair_recall_floor_passed": bool(selector.get("pair_recall_floor_passed", False)),
                    "selected_by_preselector": selected_by_preselector.get(id(candidate), False),
                    "pair_count": len(pair_scores),
                    "source_family": {
                        "source_is_vacancy": base[5],
                        "source_is_energy": base[6],
                        "source_is_factorized": base[7],
                    },
                }
            )
            for budget in PARETO_BUDGETS:
                features = _metric_features(base, budget, len(pair_scores)) + extra
                rows.append(
                    {
                        "domain": "live_v131_reconstructed",
                        "group": group,
                        "candidate_index": candidate_index,
                        "segment_k": _as_int(candidate.get("segment_k", 0)),
                        "budget": int(budget),
                        "features": features,
                        "selected_pair_count": float(min(int(budget), len(pair_scores))),
                        "selected_by_preselector": selected_by_preselector.get(id(candidate), False),
                        "pair_count": len(pair_scores),
                    }
                )
    _attach_predictions(rows, spec)
    by_candidate: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_candidate[(_as_int(row.get("group")), _as_int(row.get("candidate_index")))].append(row)
    reconstructed_candidate_picks = [
        _pick_recall_floor(candidate_rows, 0.6)
        for _, candidate_rows in sorted(by_candidate.items())
    ]
    reconstructed_policy_picks = [_pick_recall_floor(group_rows, 0.6) for _, group_rows in sorted(_group_rows(rows).items())]
    return rows, recorded_candidates, reconstructed_candidate_picks, reconstructed_policy_picks


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = {
        target: _metric_summary([_as_float(row.get("predictions", {}).get(target, 0.0)) for row in rows])
        for target in PARETO_TARGETS
    }
    features_by_name: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(FEATURE_NAMES):
        values = [_as_float(row.get("features", [])[idx], 0.0) for row in rows if len(row.get("features", [])) > idx]
        features_by_name[name] = _metric_summary(values)
    return {
        "row_count": len(rows),
        "budget_histogram": _hist([_as_int(row.get("budget", 0)) for row in rows]),
        "avg_budget": _mean([_as_float(row.get("budget", 0.0)) for row in rows]),
        "avg_selected_pair_count": _mean([_as_float(row.get("selected_pair_count", row.get("budget", 0.0))) for row in rows]),
        "pair_recall_gt_1_count": sum(1 for row in rows if _as_float(row.get("predictions", {}).get("pair_recall", 0.0)) > 1.0),
        "pair_recall_lt_0_count": sum(1 for row in rows if _as_float(row.get("predictions", {}).get("pair_recall", 0.0)) < 0.0),
        "predictions": predictions,
        "selected_by_preselector_mean": _mean([_as_float(row.get("features", [0, 0, 0, 0, 0])[4], 0.0) for row in rows if len(row.get("features", [])) > 4]),
        "feature_stats": features_by_name,
    }


def _feature_drift(offline_rows: list[dict[str, Any]], live_rows: list[dict[str, Any]], *, top_n: int = 12) -> list[dict[str, Any]]:
    drift: list[dict[str, Any]] = []
    for idx, name in enumerate(FEATURE_NAMES):
        off_values = [_as_float(row.get("features", [])[idx], 0.0) for row in offline_rows if len(row.get("features", [])) > idx]
        live_values = [_as_float(row.get("features", [])[idx], 0.0) for row in live_rows if len(row.get("features", [])) > idx]
        off_mean = _mean(off_values)
        live_mean = _mean(live_values)
        pooled = math.sqrt((_std(off_values) ** 2 + _std(live_values) ** 2) / 2.0)
        drift.append(
            {
                "feature_index": idx,
                "feature": name,
                "offline_mean": off_mean,
                "live_mean": live_mean,
                "mean_delta": live_mean - off_mean,
                "abs_delta": abs(live_mean - off_mean),
                "standardized_delta": (live_mean - off_mean) / max(pooled, 1e-12),
                "offline_p50": _quantile(off_values, 0.5),
                "live_p50": _quantile(live_values, 0.5),
            }
        )
    return sorted(drift, key=lambda item: abs(float(item["standardized_delta"])), reverse=True)[:top_n]


def _recorded_vs_reconstructed(
    recorded: list[dict[str, Any]],
    reconstructed_picks: list[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (_as_int(row.get("group")), _as_int(row.get("candidate_index"))): row
        for row in reconstructed_picks
    }
    budget_abs_errors: list[float] = []
    pair_recall_abs_errors: list[float] = []
    matched = 0
    for row in recorded:
        key = (_as_int(row.get("group")), _as_int(row.get("candidate_index")))
        candidate_pick = by_key.get(key)
        if candidate_pick is None:
            continue
        matched += 1
        budget_abs_errors.append(abs(_as_float(row.get("budget", 0.0)) - _as_float(candidate_pick.get("budget", 0.0))))
        pair_recall_abs_errors.append(
            abs(
                _as_float(row.get("predictions", {}).get("pair_recall", 0.0))
                - _as_float(candidate_pick.get("predictions", {}).get("pair_recall", 0.0))
            )
        )
    return {
        "matched_candidate_count": matched,
        "recorded_candidate_count": len(recorded),
        "budget_abs_error_mean": _mean(budget_abs_errors),
        "pair_recall_abs_error_mean": _mean(pair_recall_abs_errors),
        "note": (
            "This compares compact-output reconstruction with live selector records. "
            "Non-zero error means compact output does not preserve every full-candidate feature exactly."
        ),
    }


def run(
    *,
    v119_candidate_jsonl: Path,
    v115_candidate_jsonl: Path,
    v104_candidate_jsonl: Path,
    v130_spec_json: Path,
    v130_summary_json: Path,
    v130_choices_jsonl: Path,
    v131_eval_json: Path,
    v131_summary_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    spec = _load_json(v130_spec_json)
    v130_summary = _load_json(v130_summary_json)
    v131_summary = _load_json(v131_summary_json)
    eval_payload = _load_json(v131_eval_json)

    offline_rows, offline_policy_picks = _offline_rows(
        v119_candidate_jsonl=v119_candidate_jsonl,
        v115_candidate_jsonl=v115_candidate_jsonl,
        v104_candidate_jsonl=v104_candidate_jsonl,
        spec=spec,
    )
    offline_choices = _load_jsonl(v130_choices_jsonl)
    offline_choice_policy = [row for row in offline_choices if row.get("policy") == "pred_recall_floor0p6_balanced"]
    (
        live_rows,
        live_recorded_candidates,
        live_reconstructed_candidate_picks,
        live_reconstructed_policy_picks,
    ) = _live_rows(eval_payload, spec)
    live_recorded_selector_picks = []
    for _, group_rows in sorted(_group_rows(live_recorded_candidates).items()):
        live_recorded_selector_picks.append(max(group_rows, key=lambda row: (_as_float(row.get("selector_score", 0.0)), -_as_float(row.get("budget", 0.0)))))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "live_recorded_selector_candidates_v132.jsonl", live_recorded_candidates)
    _write_jsonl(output_dir / "live_reconstructed_candidate_picks_v132.jsonl", live_reconstructed_candidate_picks)
    _write_jsonl(output_dir / "live_reconstructed_policy_picks_v132.jsonl", live_reconstructed_policy_picks)

    feature_drift_all = _feature_drift(offline_rows, live_rows)
    feature_drift_policy = _feature_drift(offline_policy_picks, live_reconstructed_policy_picks)

    summary = {
        "mode": "v132 pure-python live calibration / feature-drift readonly diagnostic",
        "input_files": {
            "v119_candidate_jsonl": str(v119_candidate_jsonl),
            "v115_candidate_jsonl": str(v115_candidate_jsonl),
            "v104_candidate_jsonl": str(v104_candidate_jsonl),
            "v130_spec_json": str(v130_spec_json),
            "v130_summary_json": str(v130_summary_json),
            "v130_choices_jsonl": str(v130_choices_jsonl),
            "v131_eval_json": str(v131_eval_json),
            "v131_summary_json": str(v131_summary_json),
        },
        "doc_gap": {
            "doc_atomworld_mirror_md_missing": not Path("../doc/AtomWorld-Mirror.md").exists(),
        },
        "offline_v130_reference": {
            "summary_selected_fixed_top32": v130_summary.get("selected_fixed_top32", {}),
            "summary_pred_recall_floor0p6_balanced": v130_summary.get("pred_recall_floor0p6_balanced", {}),
            "policy_choice_records": _summary_for_rows(
                [
                    {
                        "budget": row.get("budget"),
                        "selected_pair_count": row.get("selected_pair_count"),
                        "features": [],
                        "predictions": {
                            "teacher_reward_sum": row.get("pred_teacher_reward_sum", 0.0),
                            "site_f1": row.get("pred_site_f1", 0.0),
                            "pair_precision": row.get("pred_pair_precision", 0.0),
                            "pair_recall": row.get("pred_pair_recall", 0.0),
                            "pair_f1": row.get("pred_pair_f1", 0.0),
                            "endpoint_f1": row.get("pred_endpoint_f1", 0.0),
                            "reward_norm": 0.0,
                        },
                    }
                    for row in offline_choice_policy
                ]
            ),
            "recomputed_all_rows": _summary_for_rows(offline_rows),
            "recomputed_policy_picks": _summary_for_rows(offline_policy_picks),
        },
        "live_v131_reference": {
            "summary_eval": v131_summary.get("eval", {}),
            "summary_diagnostic_integrity": v131_summary.get("diagnostic_integrity", {}),
            "recorded_candidate_best_rows": _summary_for_rows(live_recorded_candidates),
            "recorded_selector_suggested_rows": _summary_for_rows(live_recorded_selector_picks),
            "reconstructed_all_rows": _summary_for_rows(live_rows),
            "reconstructed_policy_picks": _summary_for_rows(live_reconstructed_policy_picks),
        },
        "drift": {
            "offline_policy_avg_budget": _mean([_as_float(row.get("budget", 0.0)) for row in offline_policy_picks]),
            "live_recorded_selector_avg_budget": _mean([_as_float(row.get("budget", 0.0)) for row in live_recorded_selector_picks]),
            "offline_policy_avg_pred_pair_recall": _mean([
                _as_float(row.get("predictions", {}).get("pair_recall", 0.0)) for row in offline_policy_picks
            ]),
            "live_recorded_selector_avg_pred_pair_recall": _mean([
                _as_float(row.get("predictions", {}).get("pair_recall", 0.0)) for row in live_recorded_selector_picks
            ]),
            "live_recorded_pair_recall_gt_1_count": sum(
                1 for row in live_recorded_selector_picks if _as_float(row.get("predictions", {}).get("pair_recall", 0.0)) > 1.0
            ),
            "feature_drift_all_rows_top": feature_drift_all,
            "feature_drift_policy_picks_top": feature_drift_policy,
            "compact_reconstruction_check": _recorded_vs_reconstructed(
                live_recorded_candidates,
                live_reconstructed_candidate_picks,
            ),
        },
        "current_judgement": (
            "v132 confirms live recall-floor calibration drift: v131 live selector predicts pair_recall far above "
            "the physical [0,1] range at tiny budgets, while v130 offline replay used conservative budgets. "
            "Do not apply v131 budget-to-projection or run replace/add; next step should be conservative budget "
            "lower bounds, prediction clipping, or live-calibrated pair-recall floor before closed-loop selector smoke."
        ),
    }

    (output_dir / "stage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "feature_drift_top_v132.json").write_text(
        json.dumps({"all_rows": feature_drift_all, "policy_picks": feature_drift_policy}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v119_candidate_jsonl", type=Path, default=DEFAULT_V119)
    parser.add_argument("--v115_candidate_jsonl", type=Path, default=DEFAULT_V115)
    parser.add_argument("--v104_candidate_jsonl", type=Path, default=DEFAULT_V104)
    parser.add_argument("--v130_spec_json", type=Path, default=DEFAULT_V130_SPEC)
    parser.add_argument("--v130_summary_json", type=Path, default=DEFAULT_V130_SUMMARY)
    parser.add_argument("--v130_choices_jsonl", type=Path, default=DEFAULT_V130_CHOICES)
    parser.add_argument("--v131_eval_json", type=Path, default=DEFAULT_V131_EVAL)
    parser.add_argument("--v131_summary_json", type=Path, default=DEFAULT_V131_SUMMARY)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = run(
        v119_candidate_jsonl=args.v119_candidate_jsonl,
        v115_candidate_jsonl=args.v115_candidate_jsonl,
        v104_candidate_jsonl=args.v104_candidate_jsonl,
        v130_spec_json=args.v130_spec_json,
        v130_summary_json=args.v130_summary_json,
        v130_choices_jsonl=args.v130_choices_jsonl,
        v131_eval_json=args.v131_eval_json,
        v131_summary_json=args.v131_summary_json,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
