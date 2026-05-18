from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diagnose_vacancy_pair_oracle_v95 import (
    _load_stage_metrics,
    _sample_row,
    _summarize_rows,
)


def _stage_frontier(stage_metrics: dict[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for stage_name, stage in stage_metrics.items():
        if not stage.get("exists"):
            continue
        for file_name, item in stage.get("files", {}).items():
            completed = item.get("completed_rollout_segments")
            delta = item.get("delta_e_ratio")
            expected_time = item.get("expected_time_ratio")
            tau_scale = item.get("tau_scale_ratio")
            site_f1 = item.get("selected_site_f1_mean")
            if completed is None or delta is None:
                continue
            entries.append(
                {
                    "stage": stage_name,
                    "file": file_name,
                    "completed_rollout_segments": completed,
                    "stop_reason": item.get("stop_reason"),
                    "delta_e_ratio": delta,
                    "expected_time_ratio": expected_time,
                    "tau_scale_ratio": tau_scale,
                    "selected_site_f1_mean": site_f1,
                    "projected_changed_count_mean": item.get("projected_changed_count_mean"),
                    "teacher_changed_count_mean": item.get("teacher_changed_count_mean"),
                    "planner_projection_change_source": item.get("planner_projection_change_source"),
                    "planner_projection_topk_source": item.get("planner_projection_topk_source"),
                    "planner_projection_topk_budget": item.get("planner_projection_topk_budget"),
                }
            )

    def best_by(key: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        usable = [item for item in candidates if item.get(key) is not None]
        if not usable:
            return None
        return max(usable, key=lambda item: float(item[key]))

    completed_entries = [item for item in entries if int(item.get("completed_rollout_segments") or 0) >= 20]
    high_energy_entries = [
        item
        for item in completed_entries
        if item.get("delta_e_ratio") is not None and float(item["delta_e_ratio"]) >= 0.9
    ]
    high_overlap_entries = [
        item
        for item in completed_entries
        if item.get("selected_site_f1_mean") is not None and float(item["selected_site_f1_mean"]) >= 0.25
    ]
    return {
        "entry_count": len(entries),
        "completed_entry_count": len(completed_entries),
        "high_energy_entry_count": len(high_energy_entries),
        "high_overlap_entry_count": len(high_overlap_entries),
        "best_delta_e": best_by("delta_e_ratio", completed_entries),
        "best_site_f1": best_by("selected_site_f1_mean", completed_entries),
        "best_site_f1_high_energy": best_by("selected_site_f1_mean", high_energy_entries),
        "best_delta_e_high_overlap": best_by("delta_e_ratio", high_overlap_entries),
        "entries": entries,
    }


def _interpret_v96(splits: dict[str, Any], frontier: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    val = splits.get("val", {})
    vacancy_f1 = float(val.get("vacancy_oracle_vs_final_f1_mean", 0.0))
    endpoint_f1 = float(val.get("endpoint_vs_final_f1_mean", 0.0))
    kmc_f1 = float(val.get("kmc_rollout_vs_final_f1_mean", 0.0))
    kmc_type = float(val.get("kmc_rollout_type_acc_on_vacancy_mean", 0.0))
    if vacancy_f1 >= 0.95 and endpoint_f1 >= 0.95 and kmc_f1 >= 0.99 and kmc_type >= 0.99:
        notes.append(
            "teacher cache contains a strong pair-level upper bound: endpoint/touched support covers final vacancy displacement, and KMC-oriented action-pair replay recovers the final typed diff."
        )
    best_high_energy = frontier.get("best_site_f1_high_energy")
    if isinstance(best_high_energy, dict):
        best_f1 = float(best_high_energy.get("selected_site_f1_mean") or 0.0)
        best_delta = float(best_high_energy.get("delta_e_ratio") or 0.0)
        if best_f1 < 0.25 and vacancy_f1 >= 0.95:
            notes.append(
                f"closed-loop high-energy strategies remain far below the teacher pair upper bound: best high-energy site F1 is {best_f1:.3f} at delta_e_ratio {best_delta:.3f}."
            )
    best_site = frontier.get("best_site_f1")
    if isinstance(best_site, dict):
        best_site_f1 = float(best_site.get("selected_site_f1_mean") or 0.0)
        best_site_delta = float(best_site.get("delta_e_ratio") or 0.0)
        if best_site_f1 >= 0.25 and best_site_delta < 0.7:
            notes.append(
                "the existing frontier still shows an energy/support conflict: overlap can improve only by sacrificing delta_e_ratio."
            )
    if vacancy_f1 >= 0.95:
        notes.append(
            "next training target should be a pair-level typed vacancy-displacement/listwise selector, not another site-wise decoder or scalar edge-score retune."
        )
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only v96 pair-level vacancy-displacement upper-bound diagnostic.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage-summary", action="append", default=[])
    parser.add_argument("--max-rows", type=int, default=32)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"expected dict cache, got {type(data)!r}")

    splits: dict[str, Any] = {}
    rows_by_split: dict[str, list[dict[str, float | int | str]]] = {}
    for split in ("train", "val"):
        samples = data.get(split)
        if samples is None:
            continue
        rows = [_sample_row(sample, split, idx) for idx, sample in enumerate(samples)]
        rows_by_split[split] = rows
        splits[split] = _summarize_rows(rows)

    stage_metrics = {
        Path(path).parent.name or Path(path).stem: _load_stage_metrics(Path(path))
        for path in args.stage_summary
    }
    frontier = _stage_frontier(stage_metrics)
    result = {
        "diagnostic": "v96_pair_level_vacancy_displacement_upper_bound",
        "cache": str(cache_path),
        "signature": data.get("signature", {}),
        "splits": splits,
        "stage_metrics": stage_metrics,
        "closed_loop_frontier": frontier,
        "interpretation": _interpret_v96(splits, frontier),
        "rows_preview": {
            split: rows[: max(int(args.max_rows), 0)]
            for split, rows in rows_by_split.items()
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
