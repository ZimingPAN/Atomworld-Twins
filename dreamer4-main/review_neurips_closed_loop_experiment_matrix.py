from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = ROOT / "results" / "neurips_closed_loop_matrix"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"_json_error": str(exc)}


def _get(data: dict[str, Any] | None, path: str, default: Any = None) -> Any:
    if data is None:
        return default
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:  # noqa: BLE001
        return False


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.6g}"
    return str(value)


def _row_from_run(run_dir: Path) -> dict[str, Any]:
    data = _read_json(run_dir / "eval_closed_loop.json")
    row: dict[str, Any] = {
        "name": run_dir.name,
        "run_dir": str(run_dir),
        "exists": bool(data and "_json_error" not in data),
        "json_error": _get(data, "_json_error", ""),
        "group": _infer_group(run_dir.name, data),
        "constraint_mode": _get(data, "constraint_mode"),
        "reference_mode": _get(data, "reference_mode"),
        "candidate_mode": _get(data, "candidate_mode"),
        "edit_mode": _get(data, "edit_mode"),
        "duration_source": _get(data, "duration_source"),
        "completed": _get(data, "completed_rollout_segments"),
        "requested": _get(data, "requested_rollout_segments"),
        "stop_reason": _get(data, "stop_reason"),
        "chosen_k_histogram": json.dumps(_get(data, "chosen_k_histogram", {}), ensure_ascii=False),
        "tau_log_mae": _get(data, "tau_expected.log_mae"),
        "tau_log_corr": _get(data, "tau_expected.log_corr"),
        "tau_scale_ratio": _get(data, "tau_expected.scale_ratio"),
        "expected_time_ratio": _get(data, "cumulative.expected_time_ratio"),
        "reward_corr": _get(data, "reward_sum.corr"),
        "reward_mae": _get(data, "reward_sum.mae"),
        "energy_abs_error_mean": _get(data, "physical_consistency.energy_abs_error_mean"),
        "energy_abs_error_final": _get(data, "physical_consistency.energy_abs_error_final"),
        "cu_chamfer_mean": _get(data, "physical_consistency.cu_chamfer_mean_distance_mean"),
        "vacancy_distance_mean": _get(data, "physical_consistency.vacancy_matching_mean_distance_mean"),
        "inventory_violation_l1_mean": _get(data, "physical_consistency.inventory_violation_l1_mean"),
        "candidate_inventory_delta_l1_mean": _get(data, "physical_consistency.candidate_inventory_delta_l1_mean"),
        "reachability_violation_rate_mean": _get(data, "physical_consistency.reachability_violation_rate_mean"),
        "cu_mean_degree_abs_error_mean": _get(data, "physical_consistency.cu_mean_degree_abs_error_mean"),
        "cu_largest_cluster_fraction_abs_error_mean": _get(
            data,
            "physical_consistency.cu_largest_cluster_fraction_abs_error_mean",
        ),
        "macro_steps": _get(data, "macro_efficiency.macro_steps"),
        "teacher_micro_events_replaced": _get(data, "macro_efficiency.teacher_micro_events_replaced"),
        "compression_ratio": _get(data, "macro_efficiency.compression_ratio_micro_events_per_macro_step"),
        "prediction_wall_time_mean_sec": _get(data, "macro_efficiency.prediction_wall_time_mean_sec"),
        "apply_wall_time_mean_sec": _get(data, "macro_efficiency.apply_wall_time_mean_sec"),
        "teacher_segment_wall_time_mean_sec": _get(data, "macro_efficiency.teacher_segment_wall_time_mean_sec"),
        "teacher_over_model_prediction_speedup": _get(data, "macro_efficiency.teacher_over_model_prediction_speedup"),
        "cuda_max_memory_allocated_mb": _get(data, "macro_efficiency.cuda_max_memory_allocated_mb"),
        "snapshot_path": _get(data, "snapshot_path"),
    }
    row["status"], row["issues"] = _classify(row)
    return row


def _infer_group(name: str, data: dict[str, Any] | None) -> str:
    if name.startswith("full_"):
        return "full"
    if name.startswith("strict_"):
        return "strict_constraints"
    if name.startswith("baseline_"):
        return "baseline"
    if name.startswith("abl_"):
        return "trained_ablation"
    if name.startswith("diag_"):
        return "macro_efficiency"
    return str(_get(data, "group", "unknown"))


def _classify(row: dict[str, Any]) -> tuple[str, str]:
    issues: list[str] = []
    if not row["exists"]:
        issues.append("missing_or_bad_json")
        return "rerun", ";".join(issues)
    completed = int(row["completed"] or 0)
    requested = int(row["requested"] or 0)
    if completed <= 0:
        issues.append("zero_completed")
    if requested > 0 and completed < min(20, requested) and row.get("stop_reason") not in {"noop_teacher_segment"}:
        issues.append("short_rollout")
    for key in ["tau_log_mae", "tau_scale_ratio", "energy_abs_error_mean", "cu_chamfer_mean"]:
        if row.get(key) is not None and not _finite(row[key]):
            issues.append(f"nonfinite_{key}")
    if row.get("constraint_mode") == "full":
        if _finite(row.get("inventory_violation_l1_mean")) and float(row["inventory_violation_l1_mean"]) > 1e-6:
            issues.append("full_inventory_violation")
        if _finite(row.get("reachability_violation_rate_mean")) and float(row["reachability_violation_rate_mean"]) > 1e-6:
            issues.append("full_reachability_violation")
    if _finite(row.get("tau_scale_ratio")):
        scale = float(row["tau_scale_ratio"])
        if scale < 0.5 or scale > 1.8:
            issues.append("tau_scale_outside_review_band")
    status = "ok" if not issues else ("rerun" if "missing_or_bad_json" in issues or "zero_completed" in issues else "review")
    return status, ";".join(issues)


def _load_rows(result_root: Path) -> list[dict[str, Any]]:
    rows = [_row_from_run(path) for path in sorted(result_root.iterdir()) if path.is_dir()]
    return sorted(rows, key=lambda row: (str(row["group"]), str(row["name"])))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def _write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    full_rows = [row for row in rows if row["group"] == "full" and row["status"] in {"ok", "review"}]
    strict_rows = [row for row in rows if row["group"] in {"strict_constraints", "baseline"}]
    ablation_rows = [row for row in rows if row["group"] == "trained_ablation"]
    efficiency_rows = [row for row in rows if row["group"] == "macro_efficiency"]
    lines = [
        "# NeurIPS Closed-loop Autonomous Rollout Review",
        "",
        "This file is generated from closed-loop autonomous rollout JSON files. It should be reviewed before paper use.",
        "",
    ]
    rerun = [row for row in rows if row["status"] == "rerun"]
    review = [row for row in rows if row["status"] == "review"]
    lines.extend(
        [
            f"- total runs: {len(rows)}",
            f"- rerun candidates: {len(rerun)}",
            f"- review rows: {len(review)}",
            "",
        ]
    )
    if full_rows:
        metrics = ["tau_log_mae", "expected_time_ratio", "energy_abs_error_mean", "cu_chamfer_mean", "reachability_violation_rate_mean", "inventory_violation_l1_mean"]
        lines.append("## Full Model Stability")
        lines.append("")
        lines.append("| metric | mean +/- std |")
        lines.append("| --- | --- |")
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in full_rows if _finite(row.get(metric))], dtype=np.float64)
            if values.size:
                lines.append(f"| {metric} | {values.mean():.6g} +/- {values.std(ddof=0):.6g} |")
        lines.append("")
    if strict_rows:
        lines.append("## Strict Constraint Ablations")
        lines.append("")
        lines.append(
            _markdown_table(
                strict_rows,
                [
                    "name",
                    "status",
                    "issues",
                    "constraint_mode",
                    "tau_log_mae",
                    "expected_time_ratio",
                    "energy_abs_error_mean",
                    "cu_chamfer_mean",
                    "reachability_violation_rate_mean",
                    "inventory_violation_l1_mean",
                    "candidate_inventory_delta_l1_mean",
                ],
            )
        )
        lines.append("")
    if ablation_rows:
        lines.append("## Trained Component Ablations")
        lines.append("")
        lines.append(
            _markdown_table(
                ablation_rows,
                [
                    "name",
                    "status",
                    "issues",
                    "tau_log_mae",
                    "expected_time_ratio",
                    "energy_abs_error_mean",
                    "cu_chamfer_mean",
                    "reachability_violation_rate_mean",
                    "inventory_violation_l1_mean",
                ],
            )
        )
        lines.append("")
    if efficiency_rows:
        lines.append("## Macro-step Efficiency Diagnostics")
        lines.append("")
        lines.append(
            _markdown_table(
                efficiency_rows,
                [
                    "name",
                    "status",
                    "issues",
                    "compression_ratio",
                    "tau_log_mae",
                    "energy_abs_error_mean",
                    "cu_chamfer_mean",
                    "teacher_over_model_prediction_speedup",
                    "cuda_max_memory_allocated_mb",
                ],
            )
        )
        lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- Strict constraint rows are inference-time ablations unless their name starts with `abl_`.")
    lines.append("- `no_reachability` and `no_constraints` use unrestricted/global candidate support and raw edits; they test physical safeguards, not a trained alternative method.")
    lines.append("- `no_continuous_time` uses the CTMC start-state baseline duration and should not be described as the learned duration head.")
    lines.append("- KMC remains the teacher/reference trajectory; PPO is available only as a paired macro-segment baseline in the fixed-k matrix unless a policy checkpoint is supplied.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bar(ax: plt.Axes, labels: list[str], values: list[float], title: str, ylabel: str, color: str) -> None:
    x = np.arange(len(labels))
    ax.bar(x, values, color=color, edgecolor="#2f2f2f", linewidth=0.7)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.6, alpha=0.8)


def _write_plot(rows: list[dict[str, Any]], path: Path) -> None:
    ok_rows = [row for row in rows if row["exists"]]
    if not ok_rows:
        return
    strict = [row for row in ok_rows if row["group"] in {"strict_constraints", "baseline"}]
    trained = [row for row in ok_rows if row["group"] == "trained_ablation"]
    efficiency = [row for row in ok_rows if row["group"] == "macro_efficiency"]
    full = [row for row in ok_rows if row["group"] == "full"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    if strict:
        labels = [row["name"].replace("strict_", "").replace("_seed0", "").replace("baseline_", "") for row in strict]
        _bar(
            axes[0, 0],
            labels,
            [float(row["energy_abs_error_mean"] or 0.0) for row in strict],
            "Closed-loop physical drift",
            "mean |E_model - E_KMC|",
            "#4C78A8",
        )
    if trained:
        labels = [row["name"].replace("abl_", "").replace("_seed0", "") for row in trained]
        _bar(
            axes[0, 1],
            labels,
            [float(row["tau_log_mae"] or 0.0) for row in trained],
            "Trained ablations: time",
            "tau log-MAE",
            "#F58518",
        )
    if efficiency:
        labels = [row["name"].replace("_seed0", "") for row in efficiency]
        axes[1, 0].scatter(
            [float(row["compression_ratio"] or 0.0) for row in efficiency],
            [float(row["energy_abs_error_mean"] or 0.0) for row in efficiency],
            s=80,
            color="#54A24B",
            edgecolor="#2f2f2f",
        )
        for row in efficiency:
            axes[1, 0].annotate(
                row["name"].replace("_seed0", ""),
                (float(row["compression_ratio"] or 0.0), float(row["energy_abs_error_mean"] or 0.0)),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
        axes[1, 0].set_title("Macro-step efficiency tradeoff", fontsize=10, weight="bold")
        axes[1, 0].set_xlabel("micro-events per macro-step")
        axes[1, 0].set_ylabel("mean energy drift")
        axes[1, 0].grid(color="#d8d8d8", linewidth=0.6, alpha=0.8)
    if full:
        labels = [row["name"] for row in full]
        width = 0.35
        x = np.arange(len(labels))
        axes[1, 1].bar(
            x - width / 2,
            [float(row["cu_chamfer_mean"] or 0.0) for row in full],
            width=width,
            label="Cu Chamfer",
            color="#B279A2",
            edgecolor="#2f2f2f",
            linewidth=0.7,
        )
        axes[1, 1].bar(
            x + width / 2,
            [float(row["vacancy_distance_mean"] or 0.0) for row in full],
            width=width,
            label="Vacancy match",
            color="#E45756",
            edgecolor="#2f2f2f",
            linewidth=0.7,
        )
        axes[1, 1].set_title("Full model structure drift", fontsize=10, weight="bold")
        axes[1, 1].set_ylabel("periodic lattice distance")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        axes[1, 1].legend(fontsize=8, frameon=False)
        axes[1, 1].grid(axis="y", color="#d8d8d8", linewidth=0.6, alpha=0.8)
    fig.suptitle("AtomWorld-Twins NeurIPS Closed-loop Review", fontsize=13, weight="bold")
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review NeurIPS closed-loop autonomous rollout matrix")
    parser.add_argument("--result_root", type=str, default=str(DEFAULT_RESULT_ROOT))
    args = parser.parse_args()
    result_root = Path(args.result_root)
    rows = _load_rows(result_root)
    _write_csv(rows, result_root / "summary.csv")
    _write_summary(rows, result_root / "summary.md")
    _write_plot(rows, result_root / "summary_bars.png")
    rerun = [row for row in rows if row["status"] == "rerun"]
    print(f"Reviewed {len(rows)} closed-loop runs; rerun candidates: {len(rerun)}")
    if rerun:
        for row in rerun:
            print(f"RERUN {row['name']}: {row['issues']}")
    print(f"Wrote {result_root / 'summary.csv'}")
    print(f"Wrote {result_root / 'summary.md'}")
    print(f"Wrote {result_root / 'summary_bars.png'}")


if __name__ == "__main__":
    main()
