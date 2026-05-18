#!/usr/bin/env python3
"""Read-only v104 two-branch candidate selector diagnostic.

This is a small, default-off bridge from v102/v103 diagnostics toward a real
candidate-level selector. It exports candidate-level training samples and runs a
leave-one-segment-out two-branch calibration:

* energy/site branch: teacher reward plus terminal site support.
* pair/count branch: terminal vacancy-pair precision/F1 plus support-count
  efficiency.

The script does not train a checkpoint or change planner behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_candidate_joint_selector_v103 import (
    FEATURE_NAMES,
    _as_float,
    _features_for_group,
    _fit_ridge,
    _minmax,
    _norm,
    _segment_candidates,
    _summarize_records,
)


def _attach_v104_targets(groups: list[dict[str, object]]) -> None:
    for group in groups:
        records = group["records"]
        reward_min, reward_max = _minmax(_as_float(r.get("teacher_reward_sum", 0.0)) for r in records)
        sqrt_reward_min, sqrt_reward_max = _minmax(
            _as_float(r.get("teacher_reward_per_sqrt_tau", 0.0)) for r in records
        )
        pair_count_inv = [
            1.0 / (1.0 + max(_as_float(r.get("vacancy_pair_selected_count", 0.0)), 0.0))
            for r in records
        ]
        pair_count_inv_min, pair_count_inv_max = _minmax(pair_count_inv)
        for record, pair_inv in zip(records, pair_count_inv):
            reward_norm = _norm(_as_float(record.get("teacher_reward_sum", 0.0)), reward_min, reward_max)
            sqrt_reward_norm = _norm(
                _as_float(record.get("teacher_reward_per_sqrt_tau", 0.0)),
                sqrt_reward_min,
                sqrt_reward_max,
            )
            non_noop = 0.0 if bool(record.get("teacher_is_noop", False)) else 1.0
            energy_site = (
                0.40 * reward_norm
                + 0.20 * sqrt_reward_norm
                + 0.25 * _as_float(record.get("site_f1", 0.0))
                + 0.10 * _as_float(record.get("site_count_efficiency", 0.0))
                + 0.05 * non_noop
            )
            pair_precision = (
                0.35 * _as_float(record.get("vacancy_pair_precision", 0.0))
                + 0.25 * _as_float(record.get("vacancy_pair_f1", 0.0))
                + 0.25 * _as_float(record.get("vacancy_pair_count_efficiency", 0.0))
                + 0.10 * _as_float(record.get("vacancy_pair_typed_endpoint_accuracy", 0.0))
                + 0.05 * _norm(pair_inv, pair_count_inv_min, pair_count_inv_max)
            )
            record["v104_energy_site_target"] = float(energy_site)
            record["v104_pair_precision_target"] = float(pair_precision)


def _normalise_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / max(hi - lo, 1e-12)


def _pick_by(records: list[dict[str, object]], key: str) -> dict[str, object]:
    return max(records, key=lambda item: _as_float(item.get(key, 0.0)))


def _two_branch_leave_one_segment_out(
    groups: list[dict[str, object]],
    *,
    pair_weights: list[float],
    ridge: float,
) -> dict[str, object]:
    picks_by_weight: dict[float, list[dict[str, object]]] = {float(w): [] for w in pair_weights}
    previews_by_weight: dict[float, list[dict[str, object]]] = {float(w): [] for w in pair_weights}
    energy_weights: list[np.ndarray] = []
    pair_weights_fit: list[np.ndarray] = []
    for heldout_idx, group in enumerate(groups):
        train_xs: list[np.ndarray] = []
        train_energy: list[np.ndarray] = []
        train_pair: list[np.ndarray] = []
        for idx, train_group in enumerate(groups):
            if idx == heldout_idx:
                continue
            records = train_group["records"]
            train_xs.append(_features_for_group(records))
            train_energy.append(
                np.asarray([_as_float(r.get("v104_energy_site_target", 0.0)) for r in records], dtype=np.float64)
            )
            train_pair.append(
                np.asarray([_as_float(r.get("v104_pair_precision_target", 0.0)) for r in records], dtype=np.float64)
            )
        if not train_xs:
            continue
        x_train = np.vstack(train_xs)
        energy_weight = _fit_ridge(x_train, np.concatenate(train_energy), ridge=ridge)
        pair_weight_vec = _fit_ridge(x_train, np.concatenate(train_pair), ridge=ridge)
        energy_weights.append(energy_weight)
        pair_weights_fit.append(pair_weight_vec)
        x_test = _features_for_group(group["records"])
        energy_pred = _normalise_scores(x_test @ energy_weight)
        pair_pred = _normalise_scores(x_test @ pair_weight_vec)
        for pair_weight in pair_weights:
            score = energy_pred + float(pair_weight) * pair_pred
            best_idx = int(np.argmax(score))
            pick = group["records"][best_idx]
            picks_by_weight[float(pair_weight)].append(pick)
            previews_by_weight[float(pair_weight)].append(
                {
                    "segment_index": int(group["segment_index"]),
                    "picked_segment_k": int(_as_float(pick.get("segment_k", 0.0))),
                    "score": float(score[best_idx]),
                    "energy_pred_norm": float(energy_pred[best_idx]),
                    "pair_pred_norm": float(pair_pred[best_idx]),
                    "energy_site_target": float(_as_float(pick.get("v104_energy_site_target", 0.0))),
                    "pair_precision_target": float(_as_float(pick.get("v104_pair_precision_target", 0.0))),
                    "site_f1": float(_as_float(pick.get("site_f1", 0.0))),
                    "vacancy_pair_f1": float(_as_float(pick.get("vacancy_pair_f1", 0.0))),
                    "vacancy_pair_precision": float(_as_float(pick.get("vacancy_pair_precision", 0.0))),
                    "teacher_reward_sum": float(_as_float(pick.get("teacher_reward_sum", 0.0))),
                }
            )
    mean_energy_weight = (
        np.mean(energy_weights, axis=0) if energy_weights else np.zeros((len(FEATURE_NAMES),), dtype=np.float64)
    )
    mean_pair_weight = (
        np.mean(pair_weights_fit, axis=0) if pair_weights_fit else np.zeros((len(FEATURE_NAMES),), dtype=np.float64)
    )
    return {
        "ridge": float(ridge),
        "feature_names": list(FEATURE_NAMES),
        "mean_energy_weights": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, mean_energy_weight.tolist())
        },
        "mean_pair_precision_weights": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, mean_pair_weight.tolist())
        },
        "pair_weight_sweep": {
            str(pair_weight): {
                "summary": _summarize_records(records),
                "preview": previews_by_weight[pair_weight][:20],
            }
            for pair_weight, records in picks_by_weight.items()
        },
    }


def _oracle_two_branch(
    groups: list[dict[str, object]],
    *,
    pair_weights: list[float],
) -> dict[str, object]:
    result = {}
    for pair_weight in pair_weights:
        picks = []
        for group in groups:
            records = group["records"]
            energy = np.asarray([_as_float(r.get("v104_energy_site_target", 0.0)) for r in records], dtype=np.float64)
            pair = np.asarray([_as_float(r.get("v104_pair_precision_target", 0.0)) for r in records], dtype=np.float64)
            score = _normalise_scores(energy) + float(pair_weight) * _normalise_scores(pair)
            picks.append(records[int(np.argmax(score))])
        result[str(float(pair_weight))] = _summarize_records(picks)
    return result


def _write_samples(groups: list[dict[str, object]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for group in groups:
            records = group["records"]
            features = _features_for_group(records)
            for candidate_index, (record, feature_row) in enumerate(zip(records, features)):
                item = {
                    "segment_index": int(group["segment_index"]),
                    "candidate_index": int(candidate_index),
                    "segment_k": int(_as_float(record.get("segment_k", 0.0))),
                    "selected_by_planner": bool(record.get("selected_by_planner", False)),
                    "features": {
                        name: float(value)
                        for name, value in zip(FEATURE_NAMES, feature_row.tolist())
                    },
                    "targets": {
                        "energy_site": float(_as_float(record.get("v104_energy_site_target", 0.0))),
                        "pair_precision": float(_as_float(record.get("v104_pair_precision_target", 0.0))),
                        "teacher_reward_sum": float(_as_float(record.get("teacher_reward_sum", 0.0))),
                        "site_f1": float(_as_float(record.get("site_f1", 0.0))),
                        "vacancy_pair_precision": float(_as_float(record.get("vacancy_pair_precision", 0.0))),
                        "vacancy_pair_recall": float(_as_float(record.get("vacancy_pair_recall", 0.0))),
                        "vacancy_pair_f1": float(_as_float(record.get("vacancy_pair_f1", 0.0))),
                        "vacancy_pair_count_efficiency": float(
                            _as_float(record.get("vacancy_pair_count_efficiency", 0.0))
                        ),
                        "vacancy_pair_selected_count": float(
                            _as_float(record.get("vacancy_pair_selected_count", 0.0))
                        ),
                        "vacancy_pair_teacher_count": float(
                            _as_float(record.get("vacancy_pair_teacher_count", 0.0))
                        ),
                    },
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
    return count


def run(input_path: Path, output_path: Path, samples_output: Path, pair_weights: list[float], ridge: float) -> dict[str, object]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    groups = _segment_candidates(data)
    _attach_v104_targets(groups)
    samples_count = _write_samples(groups, samples_output)
    all_records = [record for group in groups for record in group["records"]]
    selected = [
        next((record for record in group["records"] if bool(record.get("selected_by_planner", False))), group["records"][0])
        for group in groups
    ]
    pair_counts = [_as_float(record.get("vacancy_pair_selected_count", 0.0)) for record in all_records]
    result = {
        "mode": "v104_candidate_level_two_branch_selector_readonly",
        "input": str(input_path),
        "samples_output": str(samples_output),
        "samples_count": int(samples_count),
        "segment_count": int(len(groups)),
        "candidate_count": int(len(all_records)),
        "pair_selected_count_unique": sorted({float(value) for value in pair_counts}),
        "pair_selected_count_std": float(np.std(pair_counts)) if pair_counts else 0.0,
        "target_definition": {
            "energy_site": (
                "0.40 reward_norm + 0.20 reward_per_sqrt_tau_norm + 0.25 site_f1 + "
                "0.10 site_count_eff + 0.05 non_noop"
            ),
            "pair_precision": (
                "0.35 pair_precision + 0.25 pair_f1 + 0.25 pair_count_eff + "
                "0.10 typed_endpoint_acc + 0.05 selected_pair_count_compactness"
            ),
        },
        "all_candidates": _summarize_records(all_records),
        "selected_by_planner": _summarize_records(selected),
        "oracle_energy_site": _summarize_records([_pick_by(group["records"], "v104_energy_site_target") for group in groups]),
        "oracle_pair_precision": _summarize_records(
            [_pick_by(group["records"], "v104_pair_precision_target") for group in groups]
        ),
        "oracle_two_branch": _oracle_two_branch(groups, pair_weights=pair_weights),
        "leave_one_segment_out_two_branch": _two_branch_leave_one_segment_out(
            groups,
            pair_weights=pair_weights,
            ridge=ridge,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples_output", type=Path, required=True)
    parser.add_argument("--pair_weights", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()
    result = run(
        args.input,
        args.output,
        args.samples_output,
        pair_weights=[float(v) for v in args.pair_weights],
        ridge=float(args.ridge),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
