from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import eval_macro_long_trajectory as long_eval
import train_dreamer_macro_edit as mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark online KMC k-step teacher advancement against one macro world-model prediction"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--segments", type=int, default=100)
    parser.add_argument("--warmup_segments", type=int, default=5)
    parser.add_argument("--max_episode_steps_override", type=int, default=2000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--duration_source", type=str, default="model", choices=["model", "baseline"])
    parser.add_argument("--planner_tau_source", type=str, default=None, choices=["model", "baseline"])
    parser.add_argument("--planner_score_mode", type=str, default="energy_per_tau",
                        choices=["energy_per_tau", "energy_per_sqrt_tau", "energy"])
    parser.add_argument("--min_projected_changed_sites", type=int, default=2)
    return parser.parse_args()


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _choose_prediction(
    *,
    model: mod.MacroDreamerEditModel,
    env: mod.MacroKMCEnv,
    horizon_choices: list[int],
    max_seed_vacancies: int,
    max_candidate_sites: int,
    reward_scale: float,
    device: str,
    duration_source: str,
    planner_tau_source: str,
    planner_score_mode: str,
    min_projected_changed_sites: int,
    reward_prediction_source: str,
) -> dict[str, object] | None:
    candidates = [
        item
        for item in (
            long_eval._predict_candidate_for_horizon(
                model=model,
                duration_model=None,
                env=env,
                horizon_k=item_k,
                max_seed_vacancies=max_seed_vacancies,
                max_candidate_sites=max_candidate_sites,
                reward_scale=reward_scale,
                device=device,
                duration_source=duration_source,
                planner_tau_source=planner_tau_source,
                planner_score_mode=planner_score_mode,
                reward_prediction_source=reward_prediction_source,
            )
            for item_k in horizon_choices
        )
        if item is not None
    ]
    return long_eval._choose_planner_candidate(
        candidates,
        min_projected_changed_sites=min_projected_changed_sites,
    )


def _plot_summary(summary: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    segments = summary["segments"]
    macro_times = np.array([item["macro_predict_wall_s"] for item in segments], dtype=np.float64)
    teacher_times = np.array([item["teacher_kmc_wall_s"] for item in segments], dtype=np.float64)
    pred_tau = np.array([item["predicted_expected_tau"] for item in segments], dtype=np.float64)
    true_tau = np.array([item["traditional_kmc_expected_tau"] for item in segments], dtype=np.float64)
    x = np.arange(1, len(segments) + 1)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.2))
    ax1, ax2, ax3, ax4 = axes.ravel()
    colors = {"teacher": "#bd6a21", "macro": "#2f67ad", "green": "#2d8b57", "grey": "#7d8796"}

    means_ms = [1000.0 * float(np.mean(teacher_times)), 1000.0 * float(np.mean(macro_times))]
    ax1.bar([0, 1], means_ms, color=[colors["teacher"], colors["macro"]], alpha=0.85, width=0.55)
    for i, val in enumerate(means_ms):
        ax1.text(i, val * 1.03, f"{val:.2f}", ha="center", va="bottom", fontsize=7.2)
    ax1.set_xticks([0, 1], ["KMC\nk-step", "macro\nWM"])
    ax1.set_ylabel("wall time / segment (ms)")
    ax1.set_title("(a) Online wall-clock cost", loc="left", pad=5)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", color="#d9dee8", lw=0.5, alpha=0.8)

    ax2.plot(x, np.cumsum(teacher_times), color=colors["teacher"], lw=1.4, label="KMC k-step")
    ax2.plot(x, np.cumsum(macro_times), color=colors["macro"], lw=1.4, label="macro WM")
    ax2.set_xlabel("segments")
    ax2.set_ylabel("cumulative wall time (s)")
    ax2.set_title("(b) Cumulative benchmark time", loc="left", pad=5)
    ax2.legend(frameon=False, fontsize=6.8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(True, color="#d9dee8", lw=0.5, alpha=0.8)

    phys_rates = [
        float(np.sum(true_tau) / max(np.sum(teacher_times), 1e-12)),
        float(np.sum(pred_tau) / max(np.sum(macro_times), 1e-12)),
    ]
    ax3.bar([0, 1], phys_rates, color=[colors["teacher"], colors["green"]], alpha=0.85, width=0.55)
    for i, val in enumerate(phys_rates):
        ax3.text(i, val * 1.03, f"{val:.2e}", ha="center", va="bottom", fontsize=7.0)
    ax3.set_xticks([0, 1], ["KMC\nteacher", "macro\nWM"])
    ax3.set_ylabel(r"physical $\tau_{\rm exp}$ / wall second")
    ax3.set_title("(c) Physical-time throughput proxy", loc="left", pad=5)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.grid(axis="y", color="#d9dee8", lw=0.5, alpha=0.8)

    ax4.boxplot(
        [1000.0 * teacher_times, 1000.0 * macro_times],
        labels=["KMC\nk-step", "macro\nWM"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#eef1f5", "edgecolor": colors["grey"]},
        medianprops={"color": "#1f2933"},
    )
    ax4.set_ylabel("wall time / segment (ms)")
    ax4.set_title("(d) Per-segment variability", loc="left", pad=5)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.grid(axis="y", color="#d9dee8", lw=0.5, alpha=0.8)

    speedup = float(summary["wall_clock_speedup_teacher_over_macro"])
    fig.suptitle(
        f"Online benchmark: KMC k-step / macro WM wall-clock ratio = {speedup:.3f}",
        y=0.995,
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=1.0, h_pad=1.2, w_pad=1.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    planner_tau_source = args.planner_tau_source or args.duration_source
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    ckpt_args = ckpt["args"]
    reward_scale = float(ckpt_args.get("reward_scale", 1.0))
    reward_prediction_source = str(ckpt_args.get("reward_prediction_source", "raw"))
    ckpt_segment_ks = long_eval._segment_ks_from_ckpt_args(ckpt_args)
    horizon_choices = ckpt_segment_ks
    max_seed_vacancies = int(ckpt_args["max_seed_vacancies"])
    max_candidate_sites = int(ckpt_args["max_candidate_sites"])

    model = long_eval._build_model(ckpt, args.device)
    env = mod.MacroKMCEnv(
        long_eval._build_env_cfg(
            ckpt_args,
            max_episode_steps_override=args.max_episode_steps_override,
        )
    )
    env.reset()

    with torch.no_grad():
        for _ in range(max(0, int(args.warmup_segments))):
            selected = _choose_prediction(
                model=model,
                env=env,
                horizon_choices=horizon_choices,
                max_seed_vacancies=max_seed_vacancies,
                max_candidate_sites=max_candidate_sites,
                reward_scale=reward_scale,
                device=args.device,
                duration_source=args.duration_source,
                planner_tau_source=planner_tau_source,
                planner_score_mode=args.planner_score_mode,
                min_projected_changed_sites=args.min_projected_changed_sites,
                reward_prediction_source=reward_prediction_source,
            )
            if selected is None:
                break
            teacher_segment = long_eval._collect_teacher_segment(
                env,
                horizon_k=int(selected["segment_k"]),
                rng=rng,
            )
            if teacher_segment is None:
                break

    segments: list[dict[str, float | int]] = []
    with torch.no_grad():
        for idx in range(int(args.segments)):
            _sync(args.device)
            t0 = time.perf_counter()
            selected = _choose_prediction(
                model=model,
                env=env,
                horizon_choices=horizon_choices,
                max_seed_vacancies=max_seed_vacancies,
                max_candidate_sites=max_candidate_sites,
                reward_scale=reward_scale,
                device=args.device,
                duration_source=args.duration_source,
                planner_tau_source=planner_tau_source,
                planner_score_mode=args.planner_score_mode,
                min_projected_changed_sites=args.min_projected_changed_sites,
                reward_prediction_source=reward_prediction_source,
            )
            _sync(args.device)
            macro_wall = time.perf_counter() - t0
            if selected is None:
                break

            t0 = time.perf_counter()
            teacher_segment = long_eval._collect_teacher_segment(
                env,
                horizon_k=int(selected["segment_k"]),
                rng=rng,
            )
            teacher_wall = time.perf_counter() - t0
            if teacher_segment is None:
                break

            segments.append(
                {
                    "index": int(idx),
                    "segment_k": int(selected["segment_k"]),
                    "macro_predict_wall_s": float(macro_wall),
                    "teacher_kmc_wall_s": float(teacher_wall),
                    "predicted_expected_tau": float(selected["predicted_expected_tau"]),
                    "traditional_kmc_expected_tau": float(teacher_segment["tau_exp"]),
                    "predicted_reward_sum": float(selected["predicted_reward_sum"]),
                    "traditional_kmc_reward_sum": float(teacher_segment["reward_sum"]),
                    "reachability_violation": float(selected["reachability_violation"]),
                    "projected_changed_count": float(selected["projected_changed_count"]),
                }
            )

    if not segments:
        raise RuntimeError("No benchmark segments completed")

    macro_times = np.array([item["macro_predict_wall_s"] for item in segments], dtype=np.float64)
    teacher_times = np.array([item["teacher_kmc_wall_s"] for item in segments], dtype=np.float64)
    pred_tau = np.array([item["predicted_expected_tau"] for item in segments], dtype=np.float64)
    true_tau = np.array([item["traditional_kmc_expected_tau"] for item in segments], dtype=np.float64)
    chosen_ks = np.array([item["segment_k"] for item in segments], dtype=np.int64)
    teacher_micro_events = int(np.sum(chosen_ks))

    summary: dict[str, object] = {
        "mode": "online_kmc_vs_macro_prediction_wall_clock_benchmark",
        "checkpoint": str(checkpoint_path),
        "device": str(args.device),
        "seed": int(args.seed),
        "requested_segments": int(args.segments),
        "warmup_segments": int(args.warmup_segments),
        "completed_segments": int(len(segments)),
        "segment_ks": [int(k) for k in horizon_choices],
        "chosen_k_histogram": {str(int(k)): int(np.sum(chosen_ks == int(k))) for k in sorted(set(chosen_ks.tolist()))},
        "teacher_micro_events": teacher_micro_events,
        "evolutionary_resolution_compression": float(teacher_micro_events / max(len(segments), 1)),
        "macro_wall_s_total": float(np.sum(macro_times)),
        "teacher_kmc_wall_s_total": float(np.sum(teacher_times)),
        "macro_wall_s_mean": float(np.mean(macro_times)),
        "teacher_kmc_wall_s_mean": float(np.mean(teacher_times)),
        "macro_wall_s_median": float(np.median(macro_times)),
        "teacher_kmc_wall_s_median": float(np.median(teacher_times)),
        "wall_clock_speedup_teacher_over_macro": float(np.sum(teacher_times) / max(np.sum(macro_times), 1e-12)),
        "macro_segments_per_wall_second": float(len(segments) / max(np.sum(macro_times), 1e-12)),
        "teacher_kmc_segments_per_wall_second": float(len(segments) / max(np.sum(teacher_times), 1e-12)),
        "teacher_micro_events_per_wall_second": float(teacher_micro_events / max(np.sum(teacher_times), 1e-12)),
        "macro_predicted_physical_tau_per_wall_second": float(np.sum(pred_tau) / max(np.sum(macro_times), 1e-12)),
        "teacher_physical_tau_per_wall_second": float(np.sum(true_tau) / max(np.sum(teacher_times), 1e-12)),
        "predicted_expected_tau_total": float(np.sum(pred_tau)),
        "traditional_kmc_expected_tau_total": float(np.sum(true_tau)),
        "duration_source": str(args.duration_source),
        "planner_tau_source": str(planner_tau_source),
        "planner_score_mode": str(args.planner_score_mode),
        "segments": segments,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.plot:
        _plot_summary(summary, Path(args.plot))

    print(json.dumps({key: value for key, value in summary.items() if key != "segments"}, indent=2))


if __name__ == "__main__":
    main()
