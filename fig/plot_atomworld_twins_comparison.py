from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent


def resolve_log_path(result_dir: str | Path) -> Path | None:
    result_dir = Path(result_dir)
    candidates = [result_dir / "train.log", result_dir / "training_log.txt"]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot AtomWorld-Twins macro world model comparison figures"
    )
    parser.add_argument(
        "--model-eval",
        type=str,
        default=str(ROOT / "dreamer4-main" / "results" / "dreamer_macro_edit_v26_realized_qtrain" / "eval_time_alignment_realized_final.json"),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(ROOT / "dreamer4-main" / "results" / "dreamer_macro_edit_v26_realized_qtrain"),
    )
    parser.add_argument(
        "--ppo-eval",
        type=str,
        default=str(ROOT / "results" / "ppo_v9_results" / "ppo_macro_eval_val.json"),
    )
    parser.add_argument("--title-name", type=str, default="AtomWorld-Twins")
    parser.add_argument("--model-label", type=str, default="AtomWorld-Twins")
    parser.add_argument("--ppo-label", type=str, default="SwarmThinkers PPO")
    parser.add_argument(
        "--eval-output",
        type=str,
        default=str(FIG_DIR / "macro_edit_eval_comparison.png"),
    )
    parser.add_argument(
        "--time-output",
        type=str,
        default=str(FIG_DIR / "macro_edit_time_alignment.png"),
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_arrays(eval_data: dict) -> dict[str, np.ndarray]:
    samples = eval_data["all_samples"]
    return {
        "pred_tau": np.asarray([sample["predicted_tau"] for sample in samples], dtype=np.float64),
        "pred_tau_real": np.asarray(
            [
                sample.get(
                    "predicted_realized_tau",
                    sample.get("predicted_tau_realized", sample["predicted_tau"]),
                )
                for sample in samples
            ],
            dtype=np.float64,
        ),
        "true_tau_exp": np.asarray(
            [sample["traditional_kmc_expected_tau"] for sample in samples], dtype=np.float64
        ),
        "true_tau_real": np.asarray(
            [sample["traditional_kmc_realized_tau"] for sample in samples], dtype=np.float64
        ),
        "pred_reward": np.asarray([sample["predicted_reward_sum"] for sample in samples], dtype=np.float64),
        "true_reward": np.asarray(
            [sample["traditional_kmc_reward_sum"] for sample in samples], dtype=np.float64
        ),
        "pred_de": np.asarray([sample["predicted_delta_e"] for sample in samples], dtype=np.float64),
        "true_de": np.asarray(
            [sample["traditional_kmc_delta_e"] for sample in samples], dtype=np.float64
        ),
    }


def parse_val_log(result_dir: str | Path) -> dict[str, list[float]]:
    metrics_by_epoch = {
        key: []
        for key in [
            "reward_mae",
            "reward_corr",
            "tau_log_mae",
            "tau_log_corr",
            "tau_scale",
            "change_f1",
            "proj_change_f1",
            "chg_type_acc",
            "proj_chg_type_acc",
        ]
    }
    log_path = resolve_log_path(result_dir)
    if log_path is None:
        return {"epochs": [], **metrics_by_epoch}
    epoch_to_metrics: dict[int, dict[str, float]] = {}
    current_epoch: int | None = None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            epoch_match = re.search(r"\[Epoch\s+(\d+)/(\d+)\]", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            if ">>> VAL" not in line:
                continue
            if current_epoch is None:
                continue
            epoch_metrics: dict[str, float] = {}
            for key in metrics_by_epoch:
                match = re.search(rf"{key}=([\d.e+-]+)", line)
                if match:
                    epoch_metrics[key] = float(match.group(1))
            epoch_to_metrics[current_epoch] = epoch_metrics
    epochs = sorted(epoch_to_metrics)
    result: dict[str, list[float]] = {"epochs": epochs}
    for key in metrics_by_epoch:
        result[key] = [epoch_to_metrics[epoch].get(key, np.nan) for epoch in epochs]
    return result


def parse_train_loss(result_dir: str | Path) -> tuple[list[int], list[float]]:
    loss_by_epoch: dict[int, float] = {}
    log_path = resolve_log_path(result_dir)
    if log_path is None:
        return [], []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = re.search(r"\[Epoch\s+(\d+)/(\d+)\]\s+loss=([\d.]+)", line)
            if match:
                loss_by_epoch[int(match.group(1))] = float(match.group(3))
    epochs = sorted(loss_by_epoch)
    losses = [loss_by_epoch[epoch] for epoch in epochs]
    return epochs, losses


def _plot_eval_comparison(
    model_eval: dict,
    ppo_eval: dict,
    model_val: dict[str, list[float]],
    model_loss_epochs: list[int],
    model_loss: list[float],
    title_name: str,
    model_label: str,
    ppo_label: str,
    output_path: Path,
) -> None:
    model_color = "tab:green"
    ppo_color = "tab:red"
    kmc_color = "black"

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        f"{title_name}: Macro WM Training & Evaluation\n"
        "Baseline = Traditional KMC  |  2000 train / 400 val segments, k=4",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    gs = fig.add_gridspec(3, 2, hspace=0.36, wspace=0.25, top=0.93, bottom=0.05)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(model_loss_epochs, model_loss, color=model_color, linewidth=1.8)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Total Loss", fontsize=11)
    ax1.set_title(f"{model_label}: Training Loss", fontsize=13)
    ax1.grid(True, alpha=0.25)

    val_epochs = np.asarray(model_val.get("epochs", []), dtype=np.int32)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(val_epochs, model_val["tau_log_mae"], "o-", color=model_color, markersize=4, label=model_label)
    ax2.axhline(0.0, color=kmc_color, linestyle="--", linewidth=1.5, alpha=0.6, label="Traditional KMC (=0)")
    ax2.axhline(
        ppo_eval["tau_expected"]["log_mae"],
        color=ppo_color,
        linestyle=":",
        linewidth=2,
        alpha=0.9,
        label=f"{ppo_label} final={ppo_eval['tau_expected']['log_mae']:.3f}",
    )
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("tau log-MAE (lower = better)", fontsize=11)
    ax2.set_title("VAL: Time Prediction Error vs Traditional KMC", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(val_epochs, model_val["reward_corr"], "o-", color=model_color, markersize=4, label=model_label)
    ax3.axhline(1.0, color=kmc_color, linestyle="--", linewidth=1.5, alpha=0.6, label="Traditional KMC (=1)")
    ax3.axhline(
        ppo_eval["reward_sum"]["corr"],
        color=ppo_color,
        linestyle=":",
        linewidth=2,
        alpha=0.9,
        label=f"{ppo_label} final={ppo_eval['reward_sum']['corr']:.3f}",
    )
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Reward correlation with KMC", fontsize=11)
    ax3.set_title("VAL: Reward Correlation with Traditional KMC", fontsize=13)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(val_epochs, model_val["tau_log_corr"], "o-", color=model_color, markersize=4, label=model_label)
    ax4.axhline(1.0, color=kmc_color, linestyle="--", linewidth=1.5, alpha=0.6, label="Traditional KMC (=1)")
    ax4.axhline(
        ppo_eval["tau_expected"]["log_corr"],
        color=ppo_color,
        linestyle=":",
        linewidth=2,
        alpha=0.9,
        label=f"{ppo_label} final={ppo_eval['tau_expected']['log_corr']:.3f}",
    )
    ax4.set_xlabel("Epoch", fontsize=11)
    ax4.set_ylabel("tau log-correlation with KMC", fontsize=11)
    ax4.set_title("VAL: Time Prediction Correlation with Traditional KMC", fontsize=13)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)

    ax5 = fig.add_subplot(gs[2, 0])
    metric_names = [
        "tau log MAE\nlow",
        "tau log corr\nhigh",
        "tau scale\n~1",
        "reward MAE\nlow",
        "reward corr\nhigh",
    ]
    kmc_vals = [0.0, 1.0, 1.0, 0.0, 1.0]
    model_vals = [
        model_eval["tau_expected"]["log_mae"],
        model_eval["tau_expected"]["log_corr"],
        model_eval["tau_expected"]["scale_ratio"],
        model_eval["reward_sum"]["mae"],
        model_eval["reward_sum"]["corr"],
    ]
    ppo_vals = [
        ppo_eval["tau_expected"]["log_mae"],
        ppo_eval["tau_expected"]["log_corr"],
        ppo_eval["tau_expected"]["scale_ratio"],
        ppo_eval["reward_sum"]["mae"],
        ppo_eval["reward_sum"]["corr"],
    ]
    x = np.arange(len(metric_names))
    w = 0.26
    bars_kmc = ax5.bar(x - w, kmc_vals, width=w, color="0.55", alpha=0.8, label="Traditional KMC")
    bars_model = ax5.bar(x, model_vals, width=w, color=model_color, alpha=0.85, label=model_label)
    bars_ppo = ax5.bar(x + w, ppo_vals, width=w, color=ppo_color, alpha=0.85, label=ppo_label)
    for bars in [bars_kmc, bars_model, bars_ppo]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2,
                height + (0.03 if height >= 0 else -0.03),
                f"{height:.2f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=9,
                rotation=0,
            )
    ax5.set_xticks(x)
    ax5.set_xticklabels(metric_names, fontsize=10)
    ax5.set_title("Final Eval Metrics on Same 400 Val Segments", fontsize=13)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.2, axis="y")

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(val_epochs, model_val["change_f1"], "o-", color=model_color, markersize=4, label="change_f1")
    ax6.plot(val_epochs, model_val["proj_change_f1"], "s-", color="tab:orange", markersize=4, label="proj_change_f1")
    ax6.plot(val_epochs, model_val["proj_chg_type_acc"], "^-", color="tab:blue", markersize=4, label="proj_type_acc")
    ax6.text(
        0.98,
        0.08,
        f"{ppo_label}: no structure head",
        transform=ax6.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color=ppo_color,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": ppo_color},
    )
    ax6.set_xlabel("Epoch", fontsize=11)
    ax6.set_ylabel("Score (higher = better)", fontsize=11)
    ax6.set_title(f"{model_label}: Structure Prediction Quality", fontsize=13)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_time_alignment(
    model_eval: dict,
    ppo_eval: dict,
    model_arrays: dict[str, np.ndarray],
    ppo_arrays: dict[str, np.ndarray],
    title_name: str,
    model_label: str,
    ppo_label: str,
    output_path: Path,
) -> None:
    model_color = "tab:green"
    ppo_color = "tab:red"
    kmc_color = "black"

    fig = plt.figure(figsize=(18, 32))
    fig.suptitle(
        f"{title_name}: Time & Energy Alignment\n"
        "Baseline = Traditional KMC  |  400 val segments, k=4, cu=1.34%, v=0.02%",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    gs = fig.add_gridspec(6, 2, hspace=0.42, wspace=0.3, top=0.95, bottom=0.03)

    for col, (label, color, eval_data, arrays) in enumerate(
        [
            (model_label, model_color, model_eval, model_arrays),
            (ppo_label, ppo_color, ppo_eval, ppo_arrays),
        ]
    ):
        ax = fig.add_subplot(gs[0, col])
        pred_tau = arrays["pred_tau"]
        true_tau = arrays["true_tau_exp"]
        metrics = eval_data["tau_expected"]
        ax.scatter(true_tau, pred_tau, alpha=0.35, s=15, c=color, edgecolors="none")
        lo = min(true_tau.min(), pred_tau.min()) * 0.5
        hi = max(true_tau.max(), pred_tau.max()) * 2.0
        ax.plot([lo, hi], [lo, hi], "--", color=kmc_color, alpha=0.7, linewidth=2, label="y=x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Traditional KMC: E[tau]", fontsize=10)
        ax.set_ylabel(f"{label}: segment tau", fontsize=10)
        ax.set_title(
            f"{label} vs Traditional KMC\n"
            f"log_MAE={metrics['log_mae']:.3f}, log_corr={metrics['log_corr']:.3f}, scale={metrics['scale_ratio']:.2f}",
            fontsize=11,
        )
        ax.legend(fontsize=8, loc="upper left")
        ax.set_aspect("equal")

    for col, (label, color, arrays) in enumerate(
        [
            (model_label, model_color, model_arrays),
            (ppo_label, ppo_color, ppo_arrays),
        ]
    ):
        ax = fig.add_subplot(gs[1, col])
        eps = 1e-12
        log_ratio = np.log10(np.clip(arrays["pred_tau"], eps, None) / np.clip(arrays["true_tau_exp"], eps, None))
        pct_within_2x = np.mean(np.abs(log_ratio) < np.log10(2.0)) * 100.0
        pct_within_5x = np.mean(np.abs(log_ratio) < np.log10(5.0)) * 100.0
        median_ratio = float(np.median(log_ratio))
        ax.scatter(arrays["true_tau_exp"], log_ratio, alpha=0.3, s=12, c=color, edgecolors="none")
        ax.axhline(0.0, color=kmc_color, linewidth=2, linestyle="--", label="Perfect")
        ax.axhline(median_ratio, color=color, linewidth=1.5, linestyle="-", label=f"Median={10 ** median_ratio:.2f}x")
        ax.axhline(np.log10(2.0), color="0.5", linewidth=0.8, linestyle=":", alpha=0.4)
        ax.axhline(-np.log10(2.0), color="0.5", linewidth=0.8, linestyle=":", alpha=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("Traditional KMC: E[tau]", fontsize=10)
        ax.set_ylabel("log10(pred / KMC)", fontsize=10)
        ax.set_ylim(-5, 2)
        ax.set_title(
            f"{label}: Time Residual vs KMC\n"
            f"Within 2x: {pct_within_2x:.0f}%  |  Within 5x: {pct_within_5x:.0f}%",
            fontsize=11,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    eps = 1e-12
    model_log_ratio = np.log10(np.clip(model_arrays["pred_tau"], eps, None) / np.clip(model_arrays["true_tau_exp"], eps, None))
    ppo_log_ratio = np.log10(np.clip(ppo_arrays["pred_tau"], eps, None) / np.clip(ppo_arrays["true_tau_exp"], eps, None))

    ax_err = fig.add_subplot(gs[2, 0])
    bins = np.linspace(min(model_log_ratio.min(), ppo_log_ratio.min(), -5), max(model_log_ratio.max(), ppo_log_ratio.max(), 2), 80)
    ax_err.hist(model_log_ratio, bins=bins, alpha=0.5, color=model_color, density=True, label=f"{model_label} (std={np.std(model_log_ratio):.2f})")
    ax_err.hist(ppo_log_ratio, bins=bins, alpha=0.45, color=ppo_color, density=True, label=f"{ppo_label} (std={np.std(ppo_log_ratio):.2f})")
    ax_err.axvline(0.0, color=kmc_color, linewidth=2, linestyle="--", label="Traditional KMC")
    ax_err.axvline(np.mean(model_log_ratio), color=model_color, linewidth=1.5, linestyle="-")
    ax_err.axvline(np.mean(ppo_log_ratio), color=ppo_color, linewidth=1.5, linestyle="-")
    ax_err.set_xlabel("log10(segment tau / KMC tau)", fontsize=10)
    ax_err.set_ylabel("Density", fontsize=10)
    ax_err.set_title("Time Prediction Error Distribution", fontsize=11)
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.2)

    ax_box = fig.add_subplot(gs[2, 1])
    bp = ax_box.boxplot(
        [model_log_ratio, ppo_log_ratio],
        tick_labels=[model_label, ppo_label],
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black", "markersize": 6},
    )
    bp["boxes"][0].set_facecolor(model_color)
    bp["boxes"][0].set_alpha(0.35)
    bp["boxes"][1].set_facecolor(ppo_color)
    bp["boxes"][1].set_alpha(0.35)
    ax_box.axhline(0.0, color=kmc_color, linewidth=2, linestyle="--", label="Traditional KMC")
    ax_box.set_ylabel("log10(pred / KMC)", fontsize=10)
    ax_box.set_title("Time Error: Model vs PPO", fontsize=11)
    ax_box.legend(fontsize=9)
    ax_box.grid(True, alpha=0.2, axis="y")

    for col, (label, color, eval_data, arrays) in enumerate(
        [
            (model_label, model_color, model_eval, model_arrays),
            (ppo_label, ppo_color, ppo_eval, ppo_arrays),
        ]
    ):
        ax = fig.add_subplot(gs[3, col])
        pred_reward = arrays["pred_reward"]
        true_reward = arrays["true_reward"]
        metrics = eval_data["reward_sum"]
        lo = min(true_reward.min(), pred_reward.min()) - 0.5
        hi = max(true_reward.max(), pred_reward.max()) + 0.5
        ax.scatter(true_reward, pred_reward, alpha=0.35, s=14, c=color, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "--", color=kmc_color, alpha=0.65, linewidth=2, label="y=x")
        ax.set_xlabel("Traditional KMC: Reward Sum", fontsize=10)
        ax.set_ylabel(f"{label}: Reward Sum", fontsize=10)
        ax.set_title(
            f"{label} vs Traditional KMC Reward\n"
            f"MAE={metrics['mae']:.3f}, corr={metrics['corr']:.3f}",
            fontsize=11,
        )
        ax.legend(fontsize=9, loc="upper left")

    ax_real = fig.add_subplot(gs[4, :])
    true_tau = model_arrays["true_tau_exp"]
    true_de = model_arrays["true_de"]
    cum_kmc_tau = np.cumsum(true_tau)
    cum_kmc_de = np.cumsum(true_de)
    cum_model_tau = np.cumsum(model_arrays["pred_tau"])
    cum_model_de = np.cumsum(model_arrays["pred_de"])
    cum_ppo_tau = np.cumsum(ppo_arrays["pred_tau"])
    cum_ppo_de = np.cumsum(ppo_arrays["pred_de"])
    ax_real.plot(cum_kmc_tau, -cum_kmc_de, color=kmc_color, linewidth=2.6, alpha=0.85, label="Traditional KMC")
    ax_real.plot(cum_model_tau, -cum_model_de, color=model_color, linewidth=1.6, alpha=0.85, label=model_label)
    ax_real.plot(cum_ppo_tau, -cum_ppo_de, color=ppo_color, linewidth=1.6, alpha=0.85, label=ppo_label)
    for boundary_index in range(50, len(true_tau), 50):
        ax_real.axvline(cum_kmc_tau[boundary_index - 1], color="0.55", linestyle=":", linewidth=0.8, alpha=0.4)
    ax_real.set_xlabel("Cumulative Physical Time tau (s)  [original segment order]", fontsize=11)
    ax_real.set_ylabel("Cumulative -Delta_E (eV, energy descent)", fontsize=11)
    ax_real.set_title(
        "Energy vs Time: Real Trajectory Order\n"
        f"400 segments x k=4 = 1600 KMC steps, 8 rollouts, total KMC time = {cum_kmc_tau[-1]:.2f}s",
        fontsize=12,
    )
    ax_real.legend(fontsize=9, ncol=3)
    ax_real.grid(True, alpha=0.25)

    ax_sorted = fig.add_subplot(gs[5, :])
    order = np.argsort(true_tau)
    cum_kmc_tau_sorted = np.cumsum(true_tau[order])
    cum_kmc_de_sorted = np.cumsum(true_de[order])
    cum_model_tau_sorted = np.cumsum(model_arrays["pred_tau"][order])
    cum_model_de_sorted = np.cumsum(model_arrays["pred_de"][order])
    cum_ppo_tau_sorted = np.cumsum(ppo_arrays["pred_tau"][order])
    cum_ppo_de_sorted = np.cumsum(ppo_arrays["pred_de"][order])
    ax_sorted.plot(cum_kmc_tau_sorted, -cum_kmc_de_sorted, color=kmc_color, linewidth=2.6, alpha=0.85, label="Traditional KMC")
    ax_sorted.plot(cum_model_tau_sorted, -cum_kmc_de_sorted, color=model_color, linewidth=1.8, alpha=0.9, linestyle="-", label=f"{model_label} time")
    ax_sorted.plot(cum_kmc_tau_sorted, -cum_model_de_sorted, color=model_color, linewidth=1.4, alpha=0.65, linestyle="--", label=f"{model_label} energy")
    ax_sorted.plot(cum_ppo_tau_sorted, -cum_kmc_de_sorted, color=ppo_color, linewidth=1.8, alpha=0.9, linestyle="-", label=f"{ppo_label} time")
    ax_sorted.plot(cum_kmc_tau_sorted, -cum_ppo_de_sorted, color=ppo_color, linewidth=1.4, alpha=0.65, linestyle="--", label=f"{ppo_label} energy")
    total_de = cum_kmc_de_sorted[-1]
    idx_90 = int(np.searchsorted(cum_kmc_de_sorted, total_de * 0.9))
    if idx_90 < len(cum_kmc_tau_sorted):
        ax_sorted.axvline(cum_kmc_tau_sorted[idx_90], color="tab:red", linewidth=1.4, linestyle="--", alpha=0.55)
        pct_time = cum_kmc_tau_sorted[idx_90] / max(cum_kmc_tau_sorted[-1], 1e-12) * 100.0
        ax_sorted.annotate(
            f"90% energy drop\nat {pct_time:.1f}% of total time",
            xy=(cum_kmc_tau_sorted[idx_90], -cum_kmc_de_sorted[idx_90]),
            xytext=(cum_kmc_tau_sorted[idx_90] * 1.1, -cum_kmc_de_sorted[idx_90] * 0.5),
            arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 1.2},
            fontsize=9,
            color="tab:red",
        )
    ax_sorted.set_xlabel("Cumulative Physical Time tau (s)  [sorted by teacher tau]", fontsize=11)
    ax_sorted.set_ylabel("Cumulative -Delta_E (eV, energy descent)", fontsize=11)
    ax_sorted.set_title(
        "Energy vs Time: Sorted by Traditional KMC tau\n"
        "Solid = model time on KMC energy  |  Dashed = model energy on KMC time",
        fontsize=12,
    )
    ax_sorted.legend(fontsize=9, ncol=3)
    ax_sorted.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    model_eval = load_json(args.model_eval)
    ppo_eval = load_json(args.ppo_eval)
    model_arrays = extract_arrays(model_eval)
    ppo_arrays = extract_arrays(ppo_eval)
    model_val = parse_val_log(args.model_dir)
    model_loss_epochs, model_loss = parse_train_loss(args.model_dir)

    eval_output = Path(args.eval_output)
    time_output = Path(args.time_output)

    _plot_eval_comparison(
        model_eval=model_eval,
        ppo_eval=ppo_eval,
        model_val=model_val,
        model_loss_epochs=model_loss_epochs,
        model_loss=model_loss,
        title_name=args.title_name,
        model_label=args.model_label,
        ppo_label=args.ppo_label,
        output_path=eval_output,
    )
    _plot_time_alignment(
        model_eval=model_eval,
        ppo_eval=ppo_eval,
        model_arrays=model_arrays,
        ppo_arrays=ppo_arrays,
        title_name=args.title_name,
        model_label=args.model_label,
        ppo_label=args.ppo_label,
        output_path=time_output,
    )

    print("Saved eval comparison to", eval_output)
    print("Saved time alignment to", time_output)


if __name__ == "__main__":
    main()