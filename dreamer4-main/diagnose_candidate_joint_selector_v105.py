#!/usr/bin/env python3
"""Read-only v105 diverse candidate/count diagnostic.

This script consumes multiple v102-style compact candidate-joint JSON files.
The goal is to check whether re-collecting candidates under different pair
budgets and projection sources introduces enough support-count diversity for a
two-branch candidate selector to learn count calibration.

It does not train a checkpoint or change planner behavior.
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
    _segment_candidates,
    _summarize_records,
)
from diagnose_candidate_joint_selector_v104 import (
    _attach_v104_targets,
    _oracle_two_branch,
    _pick_by,
    _two_branch_leave_one_segment_out,
)


def _parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
    else:
        path = Path(spec)
        label = path.stem
    if not label:
        label = path.stem
    return label, path


def _load_groups(input_specs: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    groups: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for source_index, spec in enumerate(input_specs):
        label, path = _parse_input_spec(spec)
        data = json.loads(path.read_text(encoding="utf-8"))
        source_groups = _segment_candidates(data)
        source_record_count = 0
        for group in source_groups:
            group_index = len(groups)
            records: list[dict[str, object]] = []
            for record in group["records"]:
                item = dict(record)
                item["source_label"] = label
                item["source_index"] = int(source_index)
                item["source_segment_index"] = int(group.get("segment_index", len(records)))
                item["segment_index"] = int(group_index)
                records.append(item)
            source_record_count += len(records)
            groups.append(
                {
                    "segment_index": int(group_index),
                    "source_label": label,
                    "source_index": int(source_index),
                    "source_segment_index": int(group.get("segment_index", group_index)),
                    "records": records,
                }
            )
        sources.append(
            {
                "label": label,
                "path": str(path),
                "segment_count": int(len(source_groups)),
                "candidate_count": int(source_record_count),
            }
        )
    return groups, sources


def _selected_records(groups: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for group in groups:
        records = group["records"]
        if not records:
            continue
        selected.append(
            next(
                (record for record in records if bool(record.get("selected_by_planner", False))),
                records[0],
            )
        )
    return selected


def _source_summary(groups: list[dict[str, object]]) -> dict[str, object]:
    by_source: dict[str, list[dict[str, object]]] = {}
    for group in groups:
        source = str(group.get("source_label", "unknown"))
        by_source.setdefault(source, []).extend(group["records"])
    return {
        source: _summarize_records(records)
        for source, records in sorted(by_source.items())
    }


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
                    "source_label": str(record.get("source_label", "")),
                    "source_index": int(_as_float(record.get("source_index", 0.0))),
                    "source_segment_index": int(_as_float(record.get("source_segment_index", 0.0))),
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
                        "typed_endpoint_accuracy": float(
                            _as_float(record.get("vacancy_pair_typed_endpoint_accuracy", 0.0))
                        ),
                    },
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
    return count


def _slim(summary: dict[str, object]) -> dict[str, object]:
    keys = [
        "count",
        "avg_site_f1",
        "avg_vacancy_pair_precision",
        "avg_vacancy_pair_recall",
        "avg_vacancy_pair_f1",
        "avg_vacancy_pair_count_efficiency",
        "avg_teacher_reward_sum",
        "avg_projected_changed_count",
        "avg_vacancy_pair_selected_count",
    ]
    return {key: summary.get(key) for key in keys}


def run(
    input_specs: list[str],
    output_path: Path,
    samples_output: Path,
    pair_weights: list[float],
    ridge: float,
) -> dict[str, object]:
    groups, sources = _load_groups(input_specs)
    _attach_v104_targets(groups)
    samples_count = _write_samples(groups, samples_output)
    all_records = [record for group in groups for record in group["records"]]
    selected = _selected_records(groups)
    pair_counts = [_as_float(record.get("vacancy_pair_selected_count", 0.0)) for record in all_records]
    oracle = _oracle_two_branch(groups, pair_weights=pair_weights)
    loo = _two_branch_leave_one_segment_out(groups, pair_weights=pair_weights, ridge=ridge)
    result = {
        "mode": "v105_diverse_candidate_two_branch_readonly",
        "inputs": sources,
        "samples_output": str(samples_output),
        "samples_count": int(samples_count),
        "segment_count": int(len(groups)),
        "candidate_count": int(len(all_records)),
        "pair_selected_count_unique": sorted({float(value) for value in pair_counts}),
        "pair_selected_count_std": float(np.std(pair_counts)) if pair_counts else 0.0,
        "source_summaries": _source_summary(groups),
        "selected_by_planner": _summarize_records(selected),
        "oracle_energy_site": _summarize_records([_pick_by(group["records"], "v104_energy_site_target") for group in groups]),
        "oracle_pair_precision": _summarize_records(
            [_pick_by(group["records"], "v104_pair_precision_target") for group in groups]
        ),
        "oracle_two_branch": oracle,
        "leave_one_segment_out_two_branch": loo,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input eval JSON path, optionally label=path. May be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples_output", type=Path, required=True)
    parser.add_argument("--pair_weights", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    args = parser.parse_args()
    result = run(
        input_specs=list(args.input),
        output_path=args.output,
        samples_output=args.samples_output,
        pair_weights=[float(v) for v in args.pair_weights],
        ridge=float(args.ridge),
    )
    summary = {
        "mode": result["mode"],
        "segment_count": result["segment_count"],
        "candidate_count": result["candidate_count"],
        "samples_count": result["samples_count"],
        "pair_selected_count_unique": result["pair_selected_count_unique"],
        "pair_selected_count_std": result["pair_selected_count_std"],
        "selected_by_planner": _slim(result.get("selected_by_planner", {})),
        "oracle_two_branch_pair_weight_1": _slim(result.get("oracle_two_branch", {}).get("1.0", {})),
        "loo_two_branch_pair_weight_1": _slim(
            result.get("leave_one_segment_out_two_branch", {})
            .get("pair_weight_sweep", {})
            .get("1.0", {})
            .get("summary", {})
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
