from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PYTHON = "/home/likun/panziming/AtomWorld-Twins/conda/bin/python3"
PYTHONPATH = (
    "/home/likun/panziming/pydeps:"
    "/home/likun/panziming/AtomWorld-Twins/kmcteacher_backend:"
    "/home/likun/panziming/AtomWorld-Twins/LightZero-main:"
    "/home/likun/panziming/AtomWorld-Twins/dreamer4-main"
)


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


def _config_batch_size(config: dict[str, Any]) -> int:
    if config.get("batch_size") is not None:
        return int(config["batch_size"])
    if config.get("disable_teacher_candidate_augmentation") and int(config.get("max_candidate_sites", 0)) >= 1024:
        return 4
    return 32


def _load_expected_configs(result_root: Path) -> list[dict[str, Any]]:
    manifest = _read_json(result_root / "manifest.json")
    if manifest and isinstance(manifest.get("runs"), list):
        return list(manifest["runs"])
    configs = []
    for config_path in sorted(result_root.glob("*/run_config.json")):
        payload = _read_json(config_path)
        if payload:
            configs.append(payload)
    return configs


def _row_from_config(config: dict[str, Any], result_root: Path) -> dict[str, Any]:
    run_dir = ROOT / Path(config["run_dir"])
    metrics = _read_json(run_dir / "metrics.json")
    eval_path = run_dir / "eval_time_alignment.json"
    if not eval_path.exists():
        eval_path = run_dir / "eval_time_alignment_final.json"
    eval_data = _read_json(eval_path)
    long_candidates = sorted(run_dir.glob("eval_long_trajectory_*.json"))
    long_data = _read_json(long_candidates[-1]) if long_candidates else None

    row: dict[str, Any] = {
        "name": config.get("name"),
        "group": config.get("group"),
        "seed": config.get("seed"),
        "segment_k": config.get("segment_k"),
        "max_candidate_sites": config.get("max_candidate_sites"),
        "max_seed_vacancies": config.get("max_seed_vacancies"),
        "batch_size": _config_batch_size(config),
        "teacher_path_summary_mode": config.get("teacher_path_summary_mode"),
        "tau_supervision_mode": config.get("tau_supervision_mode"),
        "tau_weight": config.get("tau_weight"),
        "realized_tau_weight": config.get("realized_tau_weight"),
        "proj_weight": config.get("proj_weight"),
        "run_dir": str(run_dir),
        "train_log_exists": (run_dir / "train.log").exists(),
        "metrics_exists": metrics is not None and "_json_error" not in metrics,
        "eval_exists": eval_data is not None and "_json_error" not in eval_data,
        "long_exists": long_data is not None and "_json_error" not in long_data,
        "notes": config.get("notes", ""),
    }
    row.update(
        {
            "reward_corr": _get(metrics, "val.reward_corr", _get(eval_data, "reward_sum.corr")),
            "reward_mae": _get(metrics, "val.reward_mae", _get(eval_data, "reward_sum.mae")),
            "tau_log_mae": _get(metrics, "val.tau_log_mae", _get(eval_data, "tau_expected.log_mae")),
            "tau_log_corr": _get(metrics, "val.tau_log_corr", _get(eval_data, "tau_expected.log_corr")),
            "tau_scale_ratio": _get(metrics, "val.tau_scale_ratio", _get(eval_data, "tau_expected.scale_ratio")),
            "realized_tau_log_mae": _get(metrics, "val.realized_tau_log_mae", _get(eval_data, "tau_realized.log_mae")),
            "realized_tau_coverage_68": _get(
                metrics,
                "val.realized_tau_coverage_68",
                _get(eval_data, "tau_realized_distribution.coverage_68"),
            ),
            "realized_tau_coverage_95": _get(
                metrics,
                "val.realized_tau_coverage_95",
                _get(eval_data, "tau_realized_distribution.coverage_95"),
            ),
            "change_f1": _get(metrics, "val.change_f1"),
            "changed_type_acc": _get(metrics, "val.changed_type_acc"),
            "projected_change_f1": _get(metrics, "val.projected_change_f1"),
            "projected_changed_type_acc": _get(metrics, "val.projected_changed_type_acc"),
            "reachability_violation_rate": _get(metrics, "val.reachability_violation_rate"),
            "projected_global_l1": _get(metrics, "val.projected_global_l1"),
            "coverage": _get(metrics, "dataset.val.coverage", _get(eval_data, "dataset_stats.coverage")),
            "skipped_noop": _get(metrics, "dataset.val.skipped_noop", _get(eval_data, "dataset_stats.skipped_noop")),
            "skipped_uncovered": _get(
                metrics,
                "dataset.val.skipped_uncovered",
                _get(eval_data, "dataset_stats.skipped_uncovered"),
            ),
            "long_completed": _get(long_data, "completed_rollout_segments"),
            "long_requested": _get(long_data, "requested_rollout_segments"),
            "long_stop_reason": _get(long_data, "stop_reason"),
            "long_planner_enabled": _get(long_data, "planner_enabled"),
            "long_effective_min_projected_changed_sites": _get(
                long_data,
                "effective_min_projected_changed_sites",
            ),
            "long_expected_time_ratio": _get(long_data, "cumulative.expected_time_ratio"),
            "long_tau_log_mae": _get(long_data, "tau_expected.log_mae"),
            "long_tau_scale_ratio": _get(long_data, "tau_expected.scale_ratio"),
            "long_cumulative_delta_e_mae": _get(long_data, "cumulative.cumulative_delta_e_mae"),
        }
    )
    row["status"], row["issues"], row["rerun"] = _classify(row, config)
    return row


def _baseline_rows(result_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ppo_path = result_root / "baselines" / "ppo_v9_eval_val.json"
    ppo = _read_json(ppo_path)
    row = {
        "name": "baseline_ppo_v9",
        "group": "baseline",
        "status": "rerun" if not ppo or "_json_error" in ppo else "ok",
        "issues": "missing_eval" if not ppo else ("json_error" if "_json_error" in ppo else ""),
        "seed": "",
        "segment_k": _get(ppo, "segment_k", 4),
        "max_candidate_sites": _get(ppo, "cache_signature.max_candidate_sites", 128),
        "max_seed_vacancies": _get(ppo, "cache_signature.max_seed_vacancies", 8),
        "teacher_path_summary_mode": _get(ppo, "cache_signature.teacher_path_summary_mode", "stepwise"),
        "tau_supervision_mode": "",
        "tau_weight": "",
        "realized_tau_weight": "",
        "proj_weight": "",
        "run_dir": str(ppo_path.parent),
        "metrics_exists": bool(ppo and "_json_error" not in ppo),
        "eval_exists": bool(ppo and "_json_error" not in ppo),
        "long_exists": False,
        "reward_corr": _get(ppo, "reward_sum.corr"),
        "reward_mae": _get(ppo, "reward_sum.mae"),
        "tau_log_mae": _get(ppo, "tau_expected.log_mae"),
        "tau_log_corr": _get(ppo, "tau_expected.log_corr"),
        "tau_scale_ratio": _get(ppo, "tau_expected.scale_ratio"),
        "realized_tau_log_mae": _get(ppo, "tau_realized.log_mae"),
        "realized_tau_coverage_68": "",
        "realized_tau_coverage_95": "",
        "change_f1": "",
        "changed_type_acc": "",
        "projected_change_f1": "",
        "projected_changed_type_acc": "",
        "reachability_violation_rate": "",
        "projected_global_l1": "",
        "coverage": _get(ppo, "dataset_stats.coverage"),
        "skipped_noop": _get(ppo, "dataset_stats.skipped_noop"),
        "skipped_uncovered": _get(ppo, "dataset_stats.skipped_uncovered"),
        "long_completed": "",
        "long_requested": "",
        "long_stop_reason": "",
        "long_planner_enabled": "",
        "long_effective_min_projected_changed_sites": "",
        "long_expected_time_ratio": "",
        "long_tau_log_mae": "",
        "long_tau_scale_ratio": "",
        "long_cumulative_delta_e_mae": "",
        "notes": "SwarmThinkers PPO paired macro-segment baseline.",
        "rerun": not ppo or "_json_error" in ppo,
    }
    rows.append(row)
    return rows


def _classify(row: dict[str, Any], config: dict[str, Any]) -> tuple[str, str, bool]:
    issues: list[str] = []
    rerun = False
    if not row["train_log_exists"]:
        issues.append("missing_train_log")
        rerun = True
    if not row["metrics_exists"] and config.get("group") != "ood_eval":
        issues.append("missing_metrics")
        rerun = True
    if not row["eval_exists"]:
        issues.append("missing_eval")
        rerun = True
    if config.get("long_segments", 0) and not row["long_exists"]:
        issues.append("missing_long")
        rerun = True
    for key in ["tau_log_mae", "tau_log_corr", "tau_scale_ratio"]:
        if row.get(key) is not None and not _finite(row[key]):
            issues.append(f"nonfinite_{key}")
            rerun = True
    if row.get("reachability_violation_rate") is not None and _finite(row["reachability_violation_rate"]):
        if float(row["reachability_violation_rate"]) > 1e-6:
            issues.append("reachability_violation")
    if row.get("projected_changed_type_acc") is not None and _finite(row["projected_changed_type_acc"]):
        if float(row["projected_changed_type_acc"]) < 0.05:
            issues.append("projected_type_acc_near_zero")
    if row.get("tau_scale_ratio") is not None and _finite(row["tau_scale_ratio"]):
        scale = float(row["tau_scale_ratio"])
        if scale < 0.5 or scale > 1.5:
            issues.append("tau_scale_large_shift")
    if row.get("tau_log_mae") is not None and _finite(row["tau_log_mae"]):
        if float(row["tau_log_mae"]) > 2.0:
            issues.append("tau_log_mae_large")
    if row.get("coverage") is not None and _finite(row["coverage"]):
        if float(row["coverage"]) < 0.8:
            issues.append("low_candidate_coverage")
    if row.get("long_completed") is not None and config.get("long_segments", 0):
        try:
            completed = int(row["long_completed"])
            requested = int(row.get("long_requested") or config["long_segments"])
            stop_reason = str(row.get("long_stop_reason") or "")
            if completed < requested * 0.8:
                if stop_reason == "noop_teacher_segment" and completed > 0:
                    # Expected terminal condition under the no-op-stop long-eval
                    # protocol. Keep the row usable; interpret it as a boundary
                    # only when the scientific metrics are weak.
                    pass
                elif stop_reason == "no_legal_planner_candidate" and config.get("group") == "multi_k_optional":
                    issues.append("no_legal_planner_candidate")
                else:
                    issues.append("long_eval_short")
                    rerun = True
        except Exception:  # noqa: BLE001
            pass
    status = "ok"
    if rerun:
        status = "rerun"
    elif issues:
        status = "review"
    return status, ";".join(issues), rerun


def _append_issue(row: dict[str, Any], issue: str, rerun: bool = False) -> None:
    issues = [item for item in str(row.get("issues", "")).split(";") if item]
    if issue not in issues:
        issues.append(issue)
    row["issues"] = ";".join(issues)
    if rerun:
        row["rerun"] = True
        row["status"] = "rerun"
    elif row.get("status") == "ok":
        row["status"] = "review"


def _apply_cross_run_checks(rows: list[dict[str, Any]]) -> None:
    full_tau = [
        float(row["tau_log_mae"])
        for row in rows
        if row.get("group") == "full" and row.get("tau_log_mae") is not None and _finite(row["tau_log_mae"])
    ]
    if not full_tau:
        return
    full_tau.sort()
    median = full_tau[len(full_tau) // 2]
    threshold = max(median * 1.5, median + 1e-9)
    for row in rows:
        if row.get("tau_log_mae") is None or not _finite(row["tau_log_mae"]):
            continue
        if float(row["tau_log_mae"]) > threshold:
            _append_issue(row, "tau_log_mae_degraded")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "name",
        "group",
        "status",
        "issues",
        "seed",
        "segment_k",
        "max_candidate_sites",
        "max_seed_vacancies",
        "batch_size",
        "teacher_path_summary_mode",
        "tau_supervision_mode",
        "tau_weight",
        "realized_tau_weight",
        "proj_weight",
        "train_log_exists",
        "reward_corr",
        "tau_log_mae",
        "tau_log_corr",
        "tau_scale_ratio",
        "realized_tau_log_mae",
        "realized_tau_coverage_68",
        "realized_tau_coverage_95",
        "change_f1",
        "changed_type_acc",
        "projected_change_f1",
        "projected_changed_type_acc",
        "reachability_violation_rate",
        "coverage",
        "skipped_noop",
        "skipped_uncovered",
        "long_completed",
        "long_requested",
        "long_stop_reason",
        "long_planner_enabled",
        "long_effective_min_projected_changed_sites",
        "long_expected_time_ratio",
        "long_tau_log_mae",
        "long_tau_scale_ratio",
        "long_cumulative_delta_e_mae",
        "run_dir",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "name",
        "group",
        "status",
        "reward_corr",
        "tau_log_mae",
        "tau_log_corr",
        "tau_scale_ratio",
        "change_f1",
        "projected_changed_type_acc",
        "reachability_violation_rate",
        "coverage",
        "issues",
    ]
    lines = ["# NeurIPS fixed-k experiment matrix review", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(field)) for field in fields) + " |")
    lines.append("")
    lines.append("## Rerun candidates")
    reruns = [row for row in rows if row.get("rerun")]
    if reruns:
        for row in reruns:
            lines.append(f"- `{row['name']}`: {row.get('issues', '')}")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(number):
        return None
    return number


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, None
    var = sum((item - mean) ** 2 for item in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _fmt_mean_std(values: list[float], lower_is_better: bool = False) -> str:
    mean, std = _mean_std(values)
    if mean is None:
        return ""
    text = f"{mean:.4g}"
    if std is not None:
        text += f" +/- {std:.2g}"
    if lower_is_better:
        return text
    return text


def _fmt_delta(value: Any, reference: Any, lower_is_better: bool) -> str:
    val = _as_float(value)
    ref = _as_float(reference)
    if val is None or ref is None:
        return ""
    delta = val - ref
    if lower_is_better:
        delta = -delta
    return f"{delta:+.3g}"


def _write_paper_tables(rows: list[dict[str, Any]], path: Path) -> None:
    metric_cols = [
        ("reward_corr", "reward corr", False),
        ("tau_log_mae", "tau log-MAE", True),
        ("tau_log_corr", "tau log-corr", False),
        ("tau_scale_ratio", "tau scale", True),
        ("change_f1", "change F1", False),
        ("projected_changed_type_acc", "proj type acc", False),
        ("reachability_violation_rate", "reach viol", True),
        ("coverage", "coverage", False),
    ]
    full_rows = [row for row in rows if row.get("group") == "full"]
    full_seed0 = next((row for row in full_rows if str(row.get("name")) == "full_seed0"), None)
    lines: list[str] = ["# Paper-facing NeurIPS Tables", ""]
    lines.append(
        "This file is generated from the experiment matrix review. Use it as a drafting aid, not as a substitute for checking run logs and flagged issues."
    )
    lines.append("")
    lines.append("## Full Model Stability")
    lines.append("")
    lines.append("| metric | full seeds mean +/- std |")
    lines.append("| --- | --- |")
    for key, label, lower_is_better in metric_cols:
        values = [_as_float(row.get(key)) for row in full_rows]
        clean = [value for value in values if value is not None]
        lines.append(f"| {label} | {_fmt_mean_std(clean, lower_is_better=lower_is_better)} |")
    lines.append("")
    if full_seed0:
        lines.append("## Component Ablations")
        lines.append("")
        fields = ["name", "status", "issues", *[label for _, label, _ in metric_cols]]
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
        for row in rows:
            if row.get("group") != "ablation":
                continue
            values = [str(row.get("name", "")), str(row.get("status", "")), str(row.get("issues", ""))]
            for key, _, lower_is_better in metric_cols:
                delta = _fmt_delta(row.get(key), full_seed0.get(key), lower_is_better=lower_is_better)
                metric = _fmt(row.get(key))
                values.append(f"{metric} ({delta})" if delta else metric)
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    lines.append("## Baselines And Robustness")
    lines.append("")
    fields = ["name", "group", "status", "issues", *[label for _, label, _ in metric_cols]]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        if row.get("group") not in {"baseline", "sensitivity", "k_diagnostic", "multi_k_optional", "ood_eval"}:
            continue
        values = [str(row.get("name", "")), str(row.get("group", "")), str(row.get("status", "")), str(row.get("issues", ""))]
        values.extend(_fmt(row.get(key)) for key, _, _ in metric_cols)
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Paper-Use Guardrails")
    lines.append("")
    lines.append("- Treat `k=2`, `k=8`, and `diag_multik_248_seed0` as diagnostics only; they do not establish a solved multi-k planner.")
    lines.append("- Long evaluations that stop at `noop_teacher_segment` follow the corrected no-op-stop protocol; interpret their completed segment count together with the reported time and energy metrics, not as an engineering rerun failure.")
    lines.append("- Treat `abl_no_proj_loss_seed0` as removal of projected-state consistency loss, not as a strict no-projection or no-inventory ablation.")
    lines.append("- The current matrix does not implement strict no-reachability, no-inventory-projection, or dense-reconstruction baselines.")
    lines.append("- Rows with `status=rerun` are engineering failures until the rerun completes; rows with `status=review` need manual interpretation before entering the paper.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        (out_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        return
    plot_rows = [row for row in rows if row.get("group") != "ood_eval" and row.get("eval_exists")]
    if not plot_rows:
        return
    names = [str(row["name"]).replace("_seed0", "").replace("abl_", "") for row in plot_rows]
    metrics = [
        ("tau_log_mae", "tau log-MAE", False),
        ("tau_scale_ratio", "tau scale ratio", True),
        ("projected_changed_type_acc", "projected type acc.", False),
        ("change_f1", "change F1", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (key, title, ratio) in zip(axes.ravel(), metrics):
        vals = [float(row[key]) if row.get(key) is not None and _finite(row[key]) else float("nan") for row in plot_rows]
        ax.bar(range(len(vals)), vals, color="#3b82f6", alpha=0.85)
        if ratio:
            ax.axhline(1.0, color="#111827", linewidth=1.0, linestyle="--")
        ax.set_title(title)
        ax.set_xticks(range(len(vals)), names, rotation=55, ha="right", fontsize=7)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_bars.png", dpi=180)
    fig.savefig(out_dir / "summary_bars.pdf")
    plt.close(fig)


def _command_from_config(config: dict[str, Any], gpu: str) -> str:
    args = [
        PYTHON,
        "train_dreamer_macro_edit.py",
        "--save_dir",
        config["run_dir"],
        "--dataset_cache",
        config["cache_path"],
        "--seed",
        str(config["seed"]),
        "--teacher_path_summary_mode",
        str(config["teacher_path_summary_mode"]),
        "--tau_supervision_mode",
        str(config["tau_supervision_mode"]),
        "--tau_weight",
        str(config["tau_weight"]),
        "--realized_tau_weight",
        str(config["realized_tau_weight"]),
        "--proj_weight",
        str(config["proj_weight"]),
        "--path_weight",
        str(config["path_weight"]),
        "--prior_edit_weight",
        str(config["prior_edit_weight"]),
        "--prior_latent_weight",
        str(config["prior_latent_weight"]),
        "--prior_reward_weight",
        str(config["prior_reward_weight"]),
        "--reward_weight",
        str(config["reward_weight"]),
        "--reward_prediction_source",
        str(config["reward_prediction_source"]),
        "--train_segments",
        str(config["train_segments"]),
        "--val_segments",
        str(config["val_segments"]),
        "--epochs",
        str(config["epochs"]),
        "--batch_size",
        str(_config_batch_size(config)),
        "--lr",
        "1e-4",
        "--eval_freq",
        "5",
        "--save_freq",
        "20",
        "--max_candidate_sites",
        str(config["max_candidate_sites"]),
        "--max_seed_vacancies",
        str(config["max_seed_vacancies"]),
        "--cu_density",
        str(config["cu_density"]),
        "--v_density",
        str(config["v_density"]),
        "--device",
        f"cuda:{gpu}",
    ]
    if config.get("segment_ks"):
        args.extend(["--segment_ks", *[str(item) for item in config["segment_ks"]]])
    else:
        args.extend(["--segment_k", str(config["segment_k"])])
    if config.get("disable_teacher_candidate_augmentation"):
        args.append("--disable_teacher_candidate_augmentation")
    if config.get("eval_only_checkpoint"):
        args.extend(["--eval_only", "--resume", str(config["eval_only_checkpoint"])])
    train = shlex.join(args)
    run_dir = shlex.quote(str(config["run_dir"]))
    eval_args = [
        PYTHON,
        "eval_macro_time_alignment.py",
        "--checkpoint",
        str(config.get("eval_only_checkpoint") or Path(config["run_dir"]) / "final_model.pt"),
        "--cache",
        str(config["cache_path"]),
        "--split",
        "val",
        "--output",
        str(Path(config["run_dir"]) / "eval_time_alignment.json"),
        "--save_all_samples",
        "--device",
        f"cuda:{gpu}",
    ]
    eval_cmd = shlex.join(eval_args)
    pieces = [
        f"mkdir -p {run_dir}",
        f"{train} 2>&1 | tee {run_dir}/rerun_train.log",
        f"{eval_cmd} 2>&1 | tee {run_dir}/rerun_eval.log",
    ]
    if int(config.get("long_segments") or 0) > 0 and not config.get("eval_only_checkpoint"):
        long_args = [
            PYTHON,
            "eval_macro_long_trajectory.py",
            "--checkpoint",
            str(Path(config["run_dir"]) / "final_model.pt"),
            "--rollout_segments",
            str(config["long_segments"]),
            "--max_episode_steps_override",
            str(config.get("long_max_episode_steps", 2000)),
            "--output",
            str(Path(config["run_dir"]) / f"eval_long_trajectory_{config['long_segments']}.json"),
            "--device",
            f"cuda:{gpu}",
        ]
        pieces.append(f"{shlex.join(long_args)} 2>&1 | tee {run_dir}/rerun_long.log")
    return " && ".join(pieces)


def _eval_only_command_from_config(
    config: dict[str, Any],
    gpu: str,
    *,
    include_time_eval: bool,
    include_long_eval: bool,
) -> str:
    run_dir = shlex.quote(str(config["run_dir"]))
    pieces = [f"mkdir -p {run_dir}"]
    checkpoint = str(config.get("eval_only_checkpoint") or Path(config["run_dir"]) / "final_model.pt")
    if include_time_eval:
        eval_args = [
            PYTHON,
            "eval_macro_time_alignment.py",
            "--checkpoint",
            checkpoint,
            "--cache",
            str(config["cache_path"]),
            "--split",
            "val",
            "--output",
            str(Path(config["run_dir"]) / "eval_time_alignment.json"),
            "--save_all_samples",
            "--device",
            f"cuda:{gpu}",
        ]
        pieces.append(f"{shlex.join(eval_args)} 2>&1 | tee {run_dir}/rerun_eval.log")
    if include_long_eval and int(config.get("long_segments") or 0) > 0 and not config.get("eval_only_checkpoint"):
        long_args = [
            PYTHON,
            "eval_macro_long_trajectory.py",
            "--checkpoint",
            checkpoint,
            "--rollout_segments",
            str(config["long_segments"]),
            "--max_episode_steps_override",
            str(config.get("long_max_episode_steps", 2000)),
            "--output",
            str(Path(config["run_dir"]) / f"eval_long_trajectory_{config['long_segments']}.json"),
            "--device",
            f"cuda:{gpu}",
        ]
        pieces.append(f"{shlex.join(long_args)} 2>&1 | tee {run_dir}/rerun_long.log")
    return " && ".join(pieces)


def _rerun_command_for_row(row: dict[str, Any], config: dict[str, Any], gpu: str) -> str:
    issues = {item for item in str(row.get("issues", "")).split(";") if item}
    checkpoint_exists = (ROOT / Path(config["run_dir"]) / "final_model.pt").exists()
    metrics_ok = bool(row.get("metrics_exists"))
    eval_ok = bool(row.get("eval_exists"))
    train_ok = bool(row.get("train_log_exists")) and checkpoint_exists and metrics_ok
    if train_ok:
        long_only_issues = {"missing_long", "long_eval_short"}
        if issues and issues.issubset(long_only_issues):
            return _eval_only_command_from_config(config, gpu, include_time_eval=False, include_long_eval=True)
        if "missing_eval" in issues and issues.issubset({"missing_eval"} | long_only_issues):
            return _eval_only_command_from_config(
                config,
                gpu,
                include_time_eval=not eval_ok,
                include_long_eval=bool(issues & long_only_issues),
            )
    return _command_from_config(config, gpu)


def _baseline_command(configs: list[dict[str, Any]], result_root: Path, gpu: str) -> str | None:
    full_seed0 = next((config for config in configs if config.get("name") == "full_seed0"), None)
    if not full_seed0:
        return None
    out_dir = result_root / "baselines"
    args = [
        PYTHON,
        "../eval_ppo_macro_segments.py",
        "--checkpoint",
        "../results/ppo_v9_results/best_model.pt",
        "--config",
        "../results/ppo_v9_results/config.json",
        "--cache",
        str(full_seed0["cache_path"]),
        "--split",
        "val",
        "--output",
        str(out_dir / "ppo_v9_eval_val.json"),
        "--save_all_samples",
        "--device",
        f"cuda:{gpu}",
    ]
    return (
        f"mkdir -p {shlex.quote(str(out_dir))}"
        f" && {shlex.join(args)} 2>&1 | tee {shlex.quote(str(out_dir / 'ppo_v9_eval_rerun.log'))}"
    )


def _write_rerun(rows: list[dict[str, Any]], configs: list[dict[str, Any]], path: Path, gpu: str) -> None:
    config_by_name = {config["name"]: config for config in configs}
    result_root = path.parent
    lines = [
        "#!/usr/bin/env bash",
        "set -u",
        f"export PYTHONPATH={shlex.quote(PYTHONPATH)}",
        "export PYTHONUNBUFFERED=1",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"cd {shlex.quote(str(ROOT))}",
        "echo 'Stop a specific process with: kill <PID>. Do not use pkill or killall.'",
    ]
    for row in rows:
        if row.get("rerun") and row["name"] in config_by_name:
            lines.append(f"echo RERUN_START {shlex.quote(str(row['name']))} date=$(date -Is)")
            lines.append(_rerun_command_for_row(row, config_by_name[row["name"]], gpu))
            lines.append(f"echo RERUN_DONE {shlex.quote(str(row['name']))} date=$(date -Is)")
    if any(row.get("rerun") and row.get("name") == "baseline_ppo_v9" for row in rows):
        baseline = _baseline_command(configs, result_root, gpu)
        if baseline:
            lines.append("echo RERUN_START baseline_ppo_v9 date=$(date -Is)")
            lines.append(baseline)
            lines.append("echo RERUN_DONE baseline_ppo_v9 date=$(date -Is)")
    text = "\n\n".join(lines) + "\n"
    if not any(row.get("rerun") for row in rows):
        text = ""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review NeurIPS fixed-k experiment matrix.")
    parser.add_argument("--result_root", type=str, default=str(Path("results/neurips_fixedk_matrix")))
    parser.add_argument("--rerun_gpu", type=str, default="3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = ROOT / args.result_root
    result_root.mkdir(parents=True, exist_ok=True)
    configs = _load_expected_configs(result_root)
    rows = [_row_from_config(config, result_root) for config in configs]
    rows.extend(_baseline_rows(result_root))
    _apply_cross_run_checks(rows)
    rows.sort(key=lambda item: (str(item.get("group")), str(item.get("name"))))
    _write_csv(rows, result_root / "summary.csv")
    _write_markdown(rows, result_root / "summary.md")
    _write_paper_tables(rows, result_root / "paper_tables.md")
    _write_rerun(rows, configs, result_root / "rerun_commands.sh", args.rerun_gpu)
    _plot(rows, result_root)
    print(f"reviewed {len(rows)} runs")
    print(f"summary: {result_root / 'summary.md'}")
    print(f"csv: {result_root / 'summary.csv'}")
    reruns = [row for row in rows if row.get("rerun")]
    if reruns:
        print("rerun candidates:")
        for row in reruns:
            print(f"  {row['name']}: {row.get('issues', '')}")
    else:
        print("rerun candidates: none")


if __name__ == "__main__":
    main()
