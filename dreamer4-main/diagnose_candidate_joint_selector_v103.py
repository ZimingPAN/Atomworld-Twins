#!/usr/bin/env python3
"""Read-only v103 candidate-level selector calibration.

This script consumes a v102 ``eval_macro_long_trajectory.py`` JSON with compact
``candidate_joint_diagnostic`` records. It does not train a checkpoint. Instead
it tests whether model-visible candidate features can learn a multi-objective
teacher-probed target before we spend GPU time on a candidate-quality head.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


METRIC_FIELDS = [
    "site_precision",
    "site_recall",
    "site_f1",
    "site_count_efficiency",
    "vacancy_pair_precision",
    "vacancy_pair_recall",
    "vacancy_pair_f1",
    "vacancy_pair_count_efficiency",
    "vacancy_pair_typed_endpoint_accuracy",
    "teacher_reward_sum",
    "teacher_reward_per_sqrt_tau",
    "model_reward_sum",
    "model_expected_tau",
    "model_noop_risk",
    "projected_changed_count",
    "teacher_changed_count",
    "vacancy_pair_selected_count",
    "vacancy_pair_teacher_count",
    "candidate_quality_score",
    "pre_oracle_selection_score",
]


FEATURE_NAMES = [
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
]


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _minmax(values: Iterable[float]) -> tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0, 1.0
    lo = min(vals)
    hi = max(vals)
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return lo, lo + 1.0
    return lo, hi


def _norm(value: float, lo: float, hi: float) -> float:
    return float((float(value) - lo) / max(float(hi) - lo, 1e-12))


def _segment_candidates(data: dict[str, object]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for segment in data.get("segments", []):
        if not isinstance(segment, dict):
            continue
        records: list[dict[str, object]] = []
        for candidate in segment.get("planner_candidates", []):
            if not isinstance(candidate, dict):
                continue
            diagnostic = candidate.get("candidate_joint_diagnostic")
            if not isinstance(diagnostic, dict):
                continue
            record = dict(diagnostic)
            record["segment_index"] = int(segment.get("index", len(groups)))
            record["selected_by_planner"] = bool(record.get("selected_by_planner", False))
            records.append(record)
        if records:
            groups.append(
                {
                    "segment_index": int(segment.get("index", len(groups))),
                    "records": records,
                }
            )
    return groups


def _attach_targets(groups: list[dict[str, object]]) -> None:
    for group in groups:
        records = group["records"]
        reward_min, reward_max = _minmax(_as_float(r.get("teacher_reward_sum", 0.0)) for r in records)
        sqrt_reward_min, sqrt_reward_max = _minmax(
            _as_float(r.get("teacher_reward_per_sqrt_tau", 0.0)) for r in records
        )
        for record in records:
            reward_norm = _norm(_as_float(record.get("teacher_reward_sum", 0.0)), reward_min, reward_max)
            sqrt_reward_norm = _norm(
                _as_float(record.get("teacher_reward_per_sqrt_tau", 0.0)),
                sqrt_reward_min,
                sqrt_reward_max,
            )
            non_noop = 0.0 if bool(record.get("teacher_is_noop", False)) else 1.0
            joint_target = (
                0.35 * reward_norm
                + 0.15 * sqrt_reward_norm
                + 0.20 * _as_float(record.get("site_f1", 0.0))
                + 0.15 * _as_float(record.get("vacancy_pair_f1", 0.0))
                + 0.075 * _as_float(record.get("site_count_efficiency", 0.0))
                + 0.075 * _as_float(record.get("vacancy_pair_count_efficiency", 0.0))
                + 0.05 * non_noop
            )
            precision_target = (
                0.30 * _as_float(record.get("site_precision", 0.0))
                + 0.20 * _as_float(record.get("site_f1", 0.0))
                + 0.20 * _as_float(record.get("vacancy_pair_precision", 0.0))
                + 0.10 * _as_float(record.get("vacancy_pair_f1", 0.0))
                + 0.10 * _as_float(record.get("site_count_efficiency", 0.0))
                + 0.10 * _as_float(record.get("vacancy_pair_count_efficiency", 0.0))
            )
            record["v103_joint_target"] = float(joint_target)
            record["v103_precision_target"] = float(precision_target)


def _features_for_group(records: list[dict[str, object]]) -> np.ndarray:
    pre_lo, pre_hi = _minmax(_as_float(r.get("pre_oracle_selection_score", 0.0)) for r in records)
    reward_lo, reward_hi = _minmax(_as_float(r.get("model_reward_sum", 0.0)) for r in records)
    delta_lo, delta_hi = _minmax(_as_float(r.get("model_delta_e", 0.0)) for r in records)
    tau_inv_values = [1.0 / max(_as_float(r.get("model_expected_tau", 0.0)), 1e-12) for r in records]
    tau_lo, tau_hi = _minmax(tau_inv_values)
    noop_inv_values = [1.0 - _as_float(r.get("model_noop_risk", 0.0)) for r in records]
    noop_lo, noop_hi = _minmax(noop_inv_values)
    quality_lo, quality_hi = _minmax(_as_float(r.get("candidate_quality_score", 0.0)) for r in records)
    proj_inv_values = [1.0 / (1.0 + max(_as_float(r.get("projected_changed_count", 0.0)), 0.0)) for r in records]
    proj_lo, proj_hi = _minmax(proj_inv_values)
    pair_inv_values = [
        1.0 / (1.0 + max(_as_float(r.get("vacancy_pair_selected_count", 0.0)), 0.0))
        for r in records
    ]
    pair_lo, pair_hi = _minmax(pair_inv_values)
    density_inv_values = [1.0 - _as_float(r.get("proposal_support_density", 0.0)) for r in records]
    density_lo, density_hi = _minmax(density_inv_values)
    k_lo, k_hi = _minmax(_as_float(r.get("segment_k", 0.0)) for r in records)
    rows = []
    for record, tau_inv, noop_inv, proj_inv, pair_inv, density_inv in zip(
        records,
        tau_inv_values,
        noop_inv_values,
        proj_inv_values,
        pair_inv_values,
        density_inv_values,
    ):
        rows.append(
            [
                1.0,
                _norm(_as_float(record.get("pre_oracle_selection_score", 0.0)), pre_lo, pre_hi),
                _norm(_as_float(record.get("model_reward_sum", 0.0)), reward_lo, reward_hi),
                _norm(_as_float(record.get("model_delta_e", 0.0)), delta_lo, delta_hi),
                _norm(tau_inv, tau_lo, tau_hi),
                _norm(noop_inv, noop_lo, noop_hi),
                _norm(_as_float(record.get("candidate_quality_score", 0.0)), quality_lo, quality_hi),
                _norm(proj_inv, proj_lo, proj_hi),
                _norm(pair_inv, pair_lo, pair_hi),
                _norm(density_inv, density_lo, density_hi),
                _norm(_as_float(record.get("segment_k", 0.0)), k_lo, k_hi),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    if x.size == 0:
        return np.zeros((len(FEATURE_NAMES),), dtype=np.float64)
    reg = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    return np.linalg.pinv(x.T @ x + reg) @ x.T @ y


def _summarize_records(records: list[dict[str, object]]) -> dict[str, float]:
    if not records:
        return {}
    summary = {
        f"avg_{field}": float(np.mean([_as_float(record.get(field, 0.0)) for record in records]))
        for field in METRIC_FIELDS
    }
    summary["avg_v103_joint_target"] = float(
        np.mean([_as_float(record.get("v103_joint_target", 0.0)) for record in records])
    )
    summary["avg_v103_precision_target"] = float(
        np.mean([_as_float(record.get("v103_precision_target", 0.0)) for record in records])
    )
    summary["count"] = int(len(records))
    return summary


def _pick_by(records: list[dict[str, object]], key: str) -> dict[str, object]:
    return max(records, key=lambda item: _as_float(item.get(key, 0.0)))


def _leave_one_segment_out(
    groups: list[dict[str, object]],
    *,
    target_key: str,
    ridge: float,
) -> dict[str, object]:
    picks: list[dict[str, object]] = []
    weights: list[np.ndarray] = []
    previews: list[dict[str, object]] = []
    for heldout_idx, group in enumerate(groups):
        train_xs: list[np.ndarray] = []
        train_ys: list[np.ndarray] = []
        for idx, train_group in enumerate(groups):
            if idx == heldout_idx:
                continue
            records = train_group["records"]
            train_xs.append(_features_for_group(records))
            train_ys.append(np.asarray([_as_float(r.get(target_key, 0.0)) for r in records], dtype=np.float64))
        if not train_xs:
            continue
        x_train = np.vstack(train_xs)
        y_train = np.concatenate(train_ys)
        weight = _fit_ridge(x_train, y_train, ridge=ridge)
        weights.append(weight)
        x_test = _features_for_group(group["records"])
        scores = x_test @ weight
        best_idx = int(np.argmax(scores))
        pick = group["records"][best_idx]
        picks.append(pick)
        previews.append(
            {
                "segment_index": int(group["segment_index"]),
                "picked_segment_k": int(_as_float(pick.get("segment_k", 0.0))),
                "predicted_score": float(scores[best_idx]),
                "target": float(_as_float(pick.get(target_key, 0.0))),
                "site_f1": float(_as_float(pick.get("site_f1", 0.0))),
                "vacancy_pair_f1": float(_as_float(pick.get("vacancy_pair_f1", 0.0))),
                "teacher_reward_sum": float(_as_float(pick.get("teacher_reward_sum", 0.0))),
            }
        )
    mean_weight = np.mean(weights, axis=0) if weights else np.zeros((len(FEATURE_NAMES),), dtype=np.float64)
    return {
        "target_key": target_key,
        "ridge": float(ridge),
        "summary": _summarize_records(picks),
        "mean_weights": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, mean_weight.tolist())
        },
        "preview": previews[:20],
    }


def run(input_path: Path, output_path: Path, ridge: float) -> dict[str, object]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    groups = _segment_candidates(data)
    _attach_targets(groups)
    all_records = [record for group in groups for record in group["records"]]
    selected = [
        next((record for record in group["records"] if bool(record.get("selected_by_planner", False))), group["records"][0])
        for group in groups
    ]
    selectors = {
        "selected_by_planner": selected,
        "oracle_v103_joint_target": [_pick_by(group["records"], "v103_joint_target") for group in groups],
        "oracle_v103_precision_target": [_pick_by(group["records"], "v103_precision_target") for group in groups],
        "oracle_site_f1": [_pick_by(group["records"], "site_f1") for group in groups],
        "oracle_teacher_reward_sum": [_pick_by(group["records"], "teacher_reward_sum") for group in groups],
        "oracle_vacancy_pair_f1": [_pick_by(group["records"], "vacancy_pair_f1") for group in groups],
    }
    result = {
        "mode": "v103_candidate_level_selector_readonly",
        "input": str(input_path),
        "segment_count": int(len(groups)),
        "candidate_count": int(len(all_records)),
        "target_definition": {
            "v103_joint_target": (
                "0.35 reward_norm + 0.15 reward_per_sqrt_tau_norm + 0.20 site_f1 + "
                "0.15 vacancy_pair_f1 + 0.075 site_count_eff + 0.075 vacancy_pair_count_eff + 0.05 non_noop"
            ),
            "v103_precision_target": (
                "0.30 site_precision + 0.20 site_f1 + 0.20 vacancy_pair_precision + "
                "0.10 vacancy_pair_f1 + 0.10 site_count_eff + 0.10 vacancy_pair_count_eff"
            ),
        },
        "all_candidates": _summarize_records(all_records),
        "selectors": {
            name: _summarize_records(records)
            for name, records in selectors.items()
        },
        "leave_one_segment_out": {
            "joint_target": _leave_one_segment_out(groups, target_key="v103_joint_target", ridge=ridge),
            "precision_target": _leave_one_segment_out(groups, target_key="v103_precision_target", ridge=ridge),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()
    result = run(args.input, args.output, ridge=float(args.ridge))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
