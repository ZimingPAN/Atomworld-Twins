from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[0]
RLKMC = ROOT.parent / "kmcteacher_backend"
LIGHTZERO = ROOT.parent / "LightZero-main"
for path in [str(ROOT), str(RLKMC), str(LIGHTZERO)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import train_dreamer_macro_edit as macro_edit


CU_TYPE = 1


def _density_percent_label(density: float) -> str:
    text = f"{100.0 * float(density):.2f}"
    return text.rstrip("0").rstrip(".")


def _run_case(task: tuple[float, int, int, int]) -> dict[str, object]:
    cu_density, seed, micro_steps, max_defects = task
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    cfg = {
        "lattice_size": (40, 40, 40),
        "max_episode_steps": micro_steps + 5,
        "max_vacancies": 64,
        "max_defects": max_defects,
        "max_shells": 16,
        "stats_dim": 10,
        "temperature": 300.0,
        "reward_scale": 10.0,
        "cu_density": float(cu_density),
        "v_density": 0.0002,
        "rlkmc_topk": 16,
        "neighbor_order": "2NN",
    }
    env = macro_edit.MacroKMCEnv(cfg)
    env.reset()
    key_steps: list[int] = []
    abs_delta_e: list[float] = []
    tau_exp = 0.0
    tau_real = 0.0
    for step in range(1, micro_steps + 1):
        action = macro_edit._sample_teacher_action(env, rng)
        if action is None:
            break
        _obs, _reward, done, info = env.step(action)
        tau_exp += float(info["expected_delta_t"])
        tau_real += float(info["delta_t"])
        abs_delta_e.append(abs(float(info["delta_E"])))
        if int(info["moving_type"]) == CU_TYPE:
            key_steps.append(step)
        if done:
            break
    return {
        "cu_density": float(cu_density),
        "seed": int(seed),
        "micro_steps_requested": int(micro_steps),
        "micro_steps_completed": int(step),
        "key_event_proxy": "Cu-vacancy exchange events (moving_type == CU_TYPE)",
        "key_steps": key_steps,
        "key_count": int(len(key_steps)),
        "key_fraction": float(len(key_steps) / max(step, 1)),
        "tau_exp_sum": float(tau_exp),
        "tau_real_sum": float(tau_real),
        "mean_abs_delta_e": float(np.mean(abs_delta_e)) if abs_delta_e else 0.0,
    }


def _plot(results: list[dict[str, object]], output_prefix: Path, micro_steps: int, bin_width: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    densities = sorted({float(item["cu_density"]) for item in results})
    seeds = sorted({int(item["seed"]) for item in results})
    seed_colors = ["#6aa6d8", "#e2a64f", "#79b77e", "#b989c8", "#d06f6f"]
    density_labels = [_density_percent_label(d) for d in densities]
    by_key = {(float(item["cu_density"]), int(item["seed"])): item for item in results}

    fig = plt.figure(figsize=(11.0, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.75], wspace=0.28)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_pos = fig.add_subplot(gs[0, 1])

    x = np.arange(len(densities))
    bar_width = min(0.72 / max(len(seeds), 1), 0.22)
    for seed_idx, seed in enumerate(seeds):
        offsets = x + (seed_idx - (len(seeds) - 1) / 2) * bar_width
        heights = [int(by_key[(d, seed)]["key_count"]) for d in densities]
        ax_bar.bar(
            offsets,
            heights,
            width=bar_width * 0.92,
            color=seed_colors[seed_idx % len(seed_colors)],
            edgecolor="#2f3440",
            linewidth=0.65,
            label=f"seed {seed}",
            alpha=0.88,
        )
    means = [np.mean([int(by_key[(d, seed)]["key_count"]) for seed in seeds]) for d in densities]
    ax_bar.plot(x, means, color="#1d2430", marker="o", linewidth=1.4, markersize=4.5, label="mean")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(density_labels, rotation=25, ha="right")
    ax_bar.set_xlabel("Cu (%)")
    ax_bar.set_ylabel("Cu-exchange key events in 1000 micro steps")
    ax_bar.set_title("Key-event count")
    ax_bar.grid(axis="y", color="#d8dde6", linestyle="--", linewidth=0.7)
    for spine in ax_bar.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)
    ax_bar.legend(frameon=False, fontsize=8, loc="upper left")

    bins = np.arange(0, micro_steps + bin_width, bin_width)
    y_base = np.arange(len(densities))[::-1]
    row_height = 0.7
    seed_offsets = np.linspace(-0.22, 0.22, max(len(seeds), 1))
    cmap = plt.cm.Blues
    max_bin_count = 1
    binned_counts: dict[tuple[float, int], np.ndarray] = {}
    for d in densities:
        for seed in seeds:
            steps = np.asarray(by_key[(d, seed)]["key_steps"], dtype=np.int32)
            counts, _ = np.histogram(steps, bins=bins)
            binned_counts[(d, seed)] = counts
            max_bin_count = max(max_bin_count, int(counts.max()) if counts.size else 0)

    for density_idx, d in enumerate(densities):
        y = y_base[density_idx]
        ax_pos.barh(y, micro_steps, left=0, height=row_height, color="#f2f4f7", edgecolor="#cbd2dc", linewidth=0.7)
        mean_counts = np.mean([binned_counts[(d, seed)] for seed in seeds], axis=0)
        for bin_idx, mean_count in enumerate(mean_counts):
            if mean_count <= 0:
                continue
            left = int(bins[bin_idx])
            width = int(bins[bin_idx + 1] - bins[bin_idx])
            color = cmap(0.25 + 0.65 * min(float(mean_count) / max(max_bin_count, 1), 1.0))
            ax_pos.barh(y, width, left=left, height=row_height, color=color, edgecolor="none", alpha=0.9)
        for seed_idx, seed in enumerate(seeds):
            steps = np.asarray(by_key[(d, seed)]["key_steps"], dtype=np.float32)
            if steps.size == 0:
                continue
            y_tick = y + seed_offsets[seed_idx]
            ax_pos.vlines(
                steps,
                y_tick - 0.075,
                y_tick + 0.075,
                color=seed_colors[seed_idx % len(seed_colors)],
                linewidth=0.65,
                alpha=0.88,
            )
        total_counts = [int(by_key[(d, seed)]["key_count"]) for seed in seeds]
        ax_pos.text(
            micro_steps + 18,
            y,
            f"{np.mean(total_counts):.1f} avg",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#2f3440",
        )

    ax_pos.set_xlim(0, micro_steps + 120)
    ax_pos.set_ylim(-0.75, len(densities) - 0.25)
    ax_pos.set_yticks(y_base)
    ax_pos.set_yticklabels(density_labels)
    ax_pos.set_xlabel("Position inside 1000-step budget")
    ax_pos.set_ylabel("Cu (%)")
    ax_pos.set_title("Where key events occur")
    ax_pos.grid(axis="x", color="#d8dde6", linestyle="--", linewidth=0.7)
    for spine in ax_pos.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)
    fig.suptitle(
        "1000-step budget: Cu-density dependence of key structural exchanges",
        fontsize=12,
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        "Diagnostic proxy: key evolution = Cu-vacancy exchange event. Colored ticks show seed-level randomness; blue bar intensity shows binned mean count.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#4c5563",
    )
    fig.subplots_adjust(bottom=0.18, top=0.86)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def _plot_randomness_map(results: list[dict[str, object]], output_prefix: Path, micro_steps: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    densities = sorted({float(item["cu_density"]) for item in results})
    seeds = sorted({int(item["seed"]) for item in results})
    by_key = {(float(item["cu_density"]), int(item["seed"])): item for item in results}
    density_colors = {
        densities[0]: "#FEDF91",
        densities[1]: "#A7C6E6",
        densities[2]: "#EAA350",
        densities[3]: "#8984BE",
    }
    fallback_colors = ["#FEDF91", "#A7C6E6", "#EAA350", "#8984BE", "#78b77b", "#d06f6f"]
    line_styles = ["solid", (0, (5.0, 2.0)), (0, (5.0, 1.5, 1.2, 1.5)), (0, (1.0, 2.0))]

    fig = plt.figure(figsize=(6.7, 2.64))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.14], wspace=0.30)
    ax_raster = fig.add_subplot(gs[0, 0])
    ax_cum = fig.add_subplot(gs[0, 1])
    count_x = micro_steps + 185
    count_separator_x = micro_steps + 62
    raster_xlim_right = micro_steps + 220
    header_y = -0.38

    row = 0
    y_positions = []
    y_labels = []
    group_centers = []
    for d_idx, density in enumerate(densities):
        group_rows = []
        color = density_colors.get(density, fallback_colors[d_idx % len(fallback_colors)])
        for seed in seeds:
            item = by_key[(density, seed)]
            steps = np.asarray(item["key_steps"], dtype=np.float32)
            y = row
            group_rows.append(y)
            y_positions.append(y)
            y_labels.append(str(seed))
            ax_raster.hlines(y, 0, micro_steps, color="#e7ebf1", linewidth=6.0, zorder=0)
            if steps.size:
                ax_raster.vlines(steps, y - 0.30, y + 0.30, color=color, linewidth=0.62, alpha=0.92, zorder=2)
            ax_raster.text(
                count_x,
                y,
                f"{int(item['key_count'])}",
                va="center",
                ha="right",
                fontsize=7.2,
                color="#2f3440",
            )
            row += 1
        group_centers.append(float(np.mean(group_rows)))
        if d_idx < len(densities) - 1:
            ax_raster.axhline(row - 0.5, color="#cbd2dc", linewidth=0.7)

    for density, center in zip(densities, group_centers):
        ax_raster.text(
            -0.245,
            center,
            _density_percent_label(density),
            transform=ax_raster.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.2,
            color="#111111",
            clip_on=False,
        )

    for boundary in [250, 500, 750]:
        ax_raster.axvline(boundary, color="black", linestyle=(0, (1.0, 3.0)), linewidth=0.62, alpha=0.30, zorder=0)

    ax_raster.axvline(count_separator_x, color="#cbd2dc", linewidth=0.6, zorder=0)
    ax_raster.set_yticks(y_positions)
    ax_raster.set_yticklabels(y_labels, fontsize=7.2)
    ax_raster.set_ylim(row - 0.35, -1.05)
    ax_raster.set_xlim(0, raster_xlim_right)
    ax_raster.set_xticks(np.arange(0, micro_steps + 1, 200))
    ax_raster.set_title("Event raster: key-event positions", fontsize=9.2)
    ax_raster.set_xlabel("Position inside 1000-step budget", fontsize=8.2)
    ax_raster.set_ylabel("")
    ax_raster.text(
        -0.245,
        header_y,
        "Cu (%)",
        transform=ax_raster.get_yaxis_transform(),
        va="bottom",
        ha="right",
        fontsize=7.2,
        color="#4c5563",
        clip_on=False,
    )
    ax_raster.text(
        -0.055,
        header_y,
        "Seed",
        transform=ax_raster.get_yaxis_transform(),
        va="bottom",
        ha="right",
        fontsize=7.2,
        color="#4c5563",
        clip_on=False,
    )
    ax_raster.text(
        count_x,
        header_y,
        "count",
        va="bottom",
        ha="right",
        fontsize=7.2,
        color="#4c5563",
    )
    for spine in ax_raster.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax_raster.tick_params(axis="both", which="major", direction="out", length=4.6, width=0.95, labelsize=7.3, colors="black")
    ax_raster.tick_params(axis="x", labelbottom=True)
    ax_raster.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_raster.tick_params(axis="x", which="minor", direction="out", length=3.0, width=0.8, colors="black")
    ax_raster.set_axisbelow(True)

    x_grid = np.arange(0, micro_steps + 1)
    for d_idx, density in enumerate(densities):
        color = density_colors.get(density, fallback_colors[d_idx % len(fallback_colors)])
        line_style = line_styles[d_idx % len(line_styles)]
        curves = []
        for seed in seeds:
            steps = np.asarray(by_key[(density, seed)]["key_steps"], dtype=np.int32)
            counts = np.searchsorted(np.sort(steps), x_grid, side="right").astype(np.float32)
            total = max(float(len(steps)), 1.0)
            curve = counts / total
            curves.append(curve)
            alpha = 0.18 if len(steps) > 0 else 0.08
            ax_cum.plot(x_grid, curve, color=color, linewidth=0.9, alpha=alpha, linestyle=line_style)
        mean_curve = np.mean(np.vstack(curves), axis=0)
        ax_cum.plot(
            x_grid,
            mean_curve,
            color=color,
            linewidth=2.15,
            linestyle=line_style,
            marker="o",
            markersize=3.8,
            markerfacecolor=color,
            markeredgecolor=color,
            markevery=100,
            label=f"Cu {_density_percent_label(density)}%",
        )

    ax_cum.set_ylim(-0.02, 1.03)
    ax_cum.set_xlim(0, micro_steps)
    ax_cum.set_xlabel("Position inside 1000-step budget", fontsize=8.2)
    ax_cum.set_ylabel("Cumulative fraction of key events", fontsize=8.2)
    ax_cum.set_title("Cumulative timing", fontsize=9.2)
    ax_cum.grid(axis="y", which="major", color="black", linestyle=(0, (1.0, 3.0)), linewidth=0.65, alpha=0.58)
    ax_cum.grid(axis="x", visible=False)
    for spine in ax_cum.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax_cum.tick_params(axis="both", which="major", direction="out", length=4.6, width=0.95, labelsize=7.3, colors="black")
    ax_cum.tick_params(axis="both", which="minor", direction="out", length=3.0, width=0.8, colors="black")
    ax_cum.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_cum.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_cum.yaxis.set_major_locator(MaxNLocator(6))
    ax_cum.set_axisbelow(True)
    ax_cum.legend(frameon=False, ncol=1, fontsize=7.0, loc="lower right")

    fig.subplots_adjust(left=0.185, right=0.985, top=0.895, bottom=0.175)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.015}
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, **save_kwargs)
    fig.savefig(output_prefix.with_suffix(".pdf"), **save_kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot key-evolution positions within a 1000-step budget across Cu densities.")
    parser.add_argument("--cu_densities", type=float, nargs="+", default=[0.0025, 0.005, 0.0134, 0.02])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--micro_steps", type=int, default=1000)
    parser.add_argument("--bin_width", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_defects", type=int, default=4096)
    parser.add_argument("--output_prefix", type=Path, default=Path("Formatting_Instructions_For_NeurIPS_2026/figtest/key_evolution_budget_by_cu_density"))
    parser.add_argument("--randomness_output_prefix", type=Path, default=Path("Formatting_Instructions_For_NeurIPS_2026/figtest/key_evolution_budget_randomness_map"))
    parser.add_argument("--json_output", type=Path, default=Path("dreamer4-main/results/key_state_diagnostics/key_evolution_budget_by_cu_density.json"))
    parser.add_argument("--reuse_existing", action="store_true", help="Reuse --json_output and only redraw the figure.")
    args = parser.parse_args()

    t0 = time.time()
    if args.reuse_existing:
        payload = json.loads(args.json_output.read_text())
        results = list(payload["results"])
        micro_steps = int(payload.get("micro_steps", args.micro_steps))
        bin_width = int(payload.get("bin_width", args.bin_width))
    else:
        tasks = [(float(d), int(seed), int(args.micro_steps), int(args.max_defects)) for d in args.cu_densities for seed in args.seeds]
        results = []
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = {pool.submit(_run_case, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    json.dumps(
                        {
                            "done": len(results),
                            "total": len(tasks),
                            "cu_density": result["cu_density"],
                            "seed": result["seed"],
                            "key_count": result["key_count"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        results.sort(key=lambda item: (float(item["cu_density"]), int(item["seed"])))
        payload = {
            "diagnostic": "key_evolution_budget_by_cu_density",
            "micro_steps": int(args.micro_steps),
            "bin_width": int(args.bin_width),
            "cu_densities": [float(d) for d in args.cu_densities],
            "seeds": [int(s) for s in args.seeds],
            "key_event_proxy": "Cu-vacancy exchange event (moving_type == CU_TYPE)",
            "results": results,
            "elapsed_sec": float(time.time() - t0),
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        micro_steps = int(args.micro_steps)
        bin_width = int(args.bin_width)
    _plot(results, args.output_prefix, micro_steps, bin_width)
    _plot_randomness_map(results, args.randomness_output_prefix, micro_steps)
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "png_output": str(args.output_prefix.with_suffix(".png")),
                "pdf_output": str(args.output_prefix.with_suffix(".pdf")),
                "randomness_png_output": str(args.randomness_output_prefix.with_suffix(".png")),
                "randomness_pdf_output": str(args.randomness_output_prefix.with_suffix(".pdf")),
                "elapsed_sec": round(time.time() - t0, 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
