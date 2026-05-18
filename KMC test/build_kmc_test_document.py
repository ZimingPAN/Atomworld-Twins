#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from kmc_latex_tools import compile_pdf, latex_escape, tex_document, tex_enumerate, tex_figure, tex_figure_note, tex_itemize, tex_metric_strip, tex_table


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
REPORTS = OUTPUTS / "reports"
TABLES = OUTPUTS / "tables"

MD_OUT = REPORTS / "kmc_test_document.md"
TEX_OUT = REPORTS / "kmc_test_document.tex"
PDF_OUT = REPORTS / "kmc_test_document.pdf"

RUN_DATE = "2026-05-18"
STATUS_LABELS = {"completed": "完成", "active": "启用"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def status_label(value: str) -> str:
    return STATUS_LABELS.get(str(value), str(value))


def stage_outcome_text(stage: str) -> str:
    outcomes = {
        "1": "测试算例、能量结果",
        "2": "性能对比表、模型调用记录",
        "3": "跨尺度数据集、演化曲线",
        "4": "效率对比表、运行时间曲线",
        "5": "成分-组织趋势表、材料设计建议",
        "6": "验收报告、测试数据、结果图表",
    }
    return outcomes.get(str(stage), "")


def fmt_temp(value: str) -> str:
    return f"{float(value):.0f} K"


def short_artifacts(value: str) -> str:
    parts = [part.strip() for part in str(value).split(";") if part.strip()]
    return "; ".join(Path(part).name for part in parts)


def lattice_size_key(label: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(x) for x in str(label).split("x"))
        if len(parts) == 3:
            return parts
    except ValueError:
        pass
    return (0, 0, 0)


def manifest_entries(extra_paths: list[Path] | None = None) -> list[dict[str, object]]:
    skip_dir_suffixes = ("_render", "render_check", "quicklook_check")
    paths = set()
    for path in OUTPUTS.rglob("*"):
        if not path.is_file() or path.name == "manifest.json":
            continue
        if any(part.endswith(skip_dir_suffixes) for part in path.parts):
            continue
        paths.add(path.resolve())
    for path in extra_paths or []:
        paths.add(path.resolve())
    entries = []
    for resolved in sorted(paths, key=lambda p: str(p.relative_to(ROOT))):
        entries.append(
            {
                "path": str(resolved.relative_to(ROOT)),
                "bytes": resolved.stat().st_size if resolved.exists() else 0,
            }
        )
    return entries


def refresh_manifest() -> int:
    entries = manifest_entries([MD_OUT, TEX_OUT, PDF_OUT])
    (OUTPUTS / "manifest.json").write_text(
        json.dumps({"files": entries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(entries)


def make_summary() -> dict[str, object]:
    energy_rows = read_csv("energy_results.csv")
    perf_rows = read_csv("performance_records.csv")
    parallel_rows = read_csv("parallel_training_display.csv")
    device_rows = read_csv("device_config.csv")
    stage_rows = read_csv("stage_completion_matrix.csv")
    trend_rows = read_csv("composition_structure_trends.csv")
    main_rows = [r for r in energy_rows if "multiscale" in str(r.get("system_type", ""))]
    lattice_rows = [r for r in energy_rows if "lattice-size scan" in str(r.get("system_type", ""))]

    optimized = [r for r in perf_rows if r["mode"].startswith("optimized")]
    speedups = [float(r["speedup_vs_baseline"]) for r in optimized]
    final_energies = [float(r["final_pair_energy_eV"]) for r in energy_rows]
    cluster_sizes = [int(r["final_cu_cluster_max"]) for r in energy_rows]
    best_energy = min(trend_rows, key=lambda r: float(r["final_pair_energy_per_site_eV"]))
    best_cluster = max(trend_rows, key=lambda r: int(r["final_cu_cluster_max"]))
    temp_values = sorted({float(r["temperature_K"]) for r in main_rows or energy_rows})
    cu_values = sorted({float(r["cu_density"]) for r in main_rows or energy_rows})
    v_values = sorted({float(r["v_density"]) for r in main_rows or energy_rows})
    lattice_values = sorted({str(r["lattice_size"]) for r in lattice_rows}, key=lattice_size_key)
    step_count = sum(int(r["steps_completed"]) for r in energy_rows)

    return {
        "energy_rows": energy_rows,
        "main_rows": main_rows,
        "lattice_rows": lattice_rows,
        "step_count": step_count,
        "perf_rows": perf_rows,
        "parallel_rows": parallel_rows,
        "device": device_rows[0],
        "stage_rows": stage_rows,
        "manifest_count": len(manifest_entries([MD_OUT, TEX_OUT, PDF_OUT])),
        "case_count": len(energy_rows),
        "temperature_values": temp_values,
        "cu_values": cu_values,
        "v_values": v_values,
        "lattice_values": lattice_values,
        "speedups": speedups,
        "energy_range": (min(final_energies), max(final_energies)),
        "cluster_range": (min(cluster_sizes), max(cluster_sizes)),
        "max_nodes": max(int(r["nodes"]) for r in parallel_rows),
        "best_energy": best_energy,
        "best_cluster": best_cluster,
    }


def build_markdown(summary: dict[str, object]) -> None:
    device = summary["device"]
    best_energy = summary["best_energy"]
    best_cluster = summary["best_cluster"]
    speedups = ", ".join(f"{v:.3f}x" for v in summary["speedups"])
    temp_values = ", ".join(f"{v:g} K" for v in summary["temperature_values"])
    cu_values = ", ".join(f"{v:g}" for v in summary["cu_values"])
    v_values = ", ".join(f"{v:g}" for v in summary["v_values"])
    lattice_values = ", ".join(summary["lattice_values"])
    steps_per_case = int(summary["step_count"]) // max(int(summary["case_count"]), 1)
    energy_min, energy_max = summary["energy_range"]
    cluster_min, cluster_max = summary["cluster_range"]
    stage_lines = "\n".join(
        f"| 阶段 {r['stage']} | {r['test_content']} | {r['acceptance_indicator']} | {stage_outcome_text(r['stage'])} |"
        for r in summary["stage_rows"]
    )
    text = f"""# 强关联材料多尺度计算 KMC 测试报告

Fe-Cu-vacancy 合金体系测试 | {RUN_DATE}

## 一、测试目标

围绕强关联材料多尺度计算测试任务，选取 Fe-Cu-vacancy 合金体系作为测试样例，验证 KMC 流程在材料能量计算、跨尺度数据生成、并行计算展示和材料设计建议方面的能力。

本次测试覆盖温度 {temp_values}，Cu density {cu_values}，vacancy density {v_values}，lattice size {lattice_values}。共形成 {summary['case_count']} 个 KMC 算例，其中温度/成分/缺陷组合 {len(summary['main_rows'])} 个，lattice size 扫描 {len(summary['lattice_rows'])} 个，每组 {steps_per_case} 个 KMC step，逐步记录 {summary['step_count']} 条，并行扩展性记录覆盖 1 到 {summary['max_nodes']} 节点。

## 二、测试体系

测试体系采用 Fe-Cu-vacancy 合金体系，包括：

1. Fe 基体；
2. Fe-Cu 溶质体系；
3. Fe-vacancy 缺陷体系；
4. Fe-Cu-vacancy 复合体系；
5. Fe-Cu 团簇演化体系。

设备统一通过 `--device` 入口传入，当前记录为 requested={device['requested_device']}，resolved={device['resolved_device']}，backend={device['backend']}，status={device['status']}。DeepH 调用路径为 `DeepHCalculator -> predict_hamiltonian -> total_energy`，DeepKS 调用路径为 `DeepKSCalculator -> get_potential_energy`。

## 三、分阶段测试内容

| 阶段 | 测试内容 | 对应考核指标 | 成果形式 |
| --- | --- | --- | --- |
{stage_lines}

## 四、主要输出结果

本测试最终形成以下结果：

1. Fe-Cu-vacancy 典型测试算例；
2. 能量计算结果表；
3. 软件适配和性能测试记录；
4. 跨尺度数据集；
5. 材料演化曲线；
6. Cu 团簇组织结构图；
7. 计算效率对比表；
8. 材料设计优化建议；
9. 项目验收报告。

| 序号 | 主要输出结果 | 对应文件 |
| --- | --- | --- |
| 1 | Fe-Cu-vacancy 典型测试算例 | `outputs/cases/typical_cases.json` |
| 2 | 能量计算结果表 | `outputs/tables/energy_results.csv` |
| 3 | 软件适配和性能测试记录 | `outputs/reports/software_adaptation_and_performance.md`; `outputs/tables/performance_records.csv`; `outputs/tables/module_timing_breakdown.csv` |
| 4 | 跨尺度数据集 | `outputs/datasets/multiscale_dataset.csv`; `outputs/tables/multiscale_dataset.csv` |
| 5 | 材料演化曲线 | `outputs/figures/material_evolution_curves.png` |
| 6 | Cu 团簇组织结构图 | `outputs/figures/cu_cluster_structure.png` |
| 7 | 计算效率对比表 | `outputs/tables/efficiency_comparison.csv`; `outputs/figures/runtime_comparison.png` |
| 8 | 材料设计优化建议 | `outputs/reports/material_design_recommendations.md` |
| 9 | 项目验收报告 | `outputs/reports/acceptance_report.pdf`; `outputs/reports/acceptance_report.tex`; `outputs/reports/acceptance_report.md` |

## 五、验收展示内容

验收时重点展示：

1. 软件适配结果：说明 Fe-Cu-vacancy 算例可以正常运行；
2. 性能优化结果：展示优化前后运行时间对比；
3. 跨尺度数据结果：展示不同温度、成分条件下的数据生成；
4. 材料演化结果：展示能量变化曲线和 Cu 团簇结构；
5. 设计优化结果：给出材料成分和组织调控建议；
6. 验收报告：汇总各阶段任务完成情况。

主要图形结果如下：

- 图 1：`outputs/figures/material_evolution_curves.png`，展示温度下能量变化、Cu density 下团簇演化、vacancy density 下物理时间演化。
  - 图示说明：左图展示不同温度条件下平均能量随 KMC step 的变化；中图展示不同 Cu density 条件下最大 Cu 团簇尺寸变化，并去除了最高 Cu density 曲线以突出低中 Cu 含量差异；右图展示不同 vacancy density 条件下物理时间推进差异。
- 图 2：`outputs/figures/cu_cluster_structure.png`，展示 Cu 团簇组织结构图。
  - 图示说明：图中选取最大 Cu 团簇增长最明显的算例，上排给出初始与最终 whole box 的 Cu 原子空间分布，下排给出局部放大区域；红色描边表示当前最大团簇，黄色点表示最终最大团簇中新加入的 Cu 位点，用于突出 initial 到 final 的团簇增长。
- 图 3：`outputs/figures/runtime_comparison.png`，展示优化前后运行时间对比。
  - 图示说明：左图使用对数坐标比较全量速率刷新 baseline 与增量更新模式的运行时间，右图给出各 lattice size 下的 measured speedup，用于展示性能优化趋势。

## 六、简要结论

通过 Fe-Cu-vacancy 测试样例，本测试展示了 KMC 从材料能量计算、跨尺度数据生成、大规模并行计算展示到材料设计建议的完整流程。测试结果可支撑强关联材料体系的数值模拟、智能计算和优化设计任务验收。

关键结果摘要如下：

| 项目 | 结果 |
| --- | --- |
| 总 KMC 算例数量 | {summary['case_count']} |
| 逐步 KMC 记录 | {summary['step_count']} |
| 能量结果行数 | {len(summary['energy_rows'])} |
| 最终 pair energy 范围 | {energy_min:.6f} 到 {energy_max:.6f} eV |
| Cu 最大团簇范围 | {cluster_min} 到 {cluster_max} |
| 十种 lattice size speedup | {speedups} |
| DeepH / DeepKS 并行扩展性记录 | 覆盖 1 到 {summary['max_nodes']} 节点 |
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text(text, encoding="utf-8")


def build_latex(summary: dict[str, object]) -> None:
    device = summary["device"]
    speedups = ", ".join(f"{v:.3f}x" for v in summary["speedups"])
    temp_values = ", ".join(f"{v:g} K" for v in summary["temperature_values"])
    cu_values = ", ".join(f"{v:g}" for v in summary["cu_values"])
    v_values = ", ".join(f"{v:g}" for v in summary["v_values"])
    lattice_values = ", ".join(summary["lattice_values"])
    steps_per_case = int(summary["step_count"]) // max(int(summary["case_count"]), 1)
    energy_min, energy_max = summary["energy_range"]
    cluster_min, cluster_max = summary["cluster_range"]

    body = "\n\n".join(
        [
            "\\section*{一、测试目标}\n"
            "围绕强关联材料多尺度计算测试任务，选取 Fe-Cu-vacancy 合金体系作为测试样例，验证 KMC 流程在材料能量计算、跨尺度数据生成、并行计算展示和材料设计建议方面的能力。\n\n"
            f"本次测试覆盖温度 {temp_values}，Cu density {cu_values}，vacancy density {v_values}，lattice size {lattice_values}。"
            f"共形成 {summary['case_count']} 个 KMC 算例，其中温度/成分/缺陷组合 {len(summary['main_rows'])} 个，lattice size 扫描 {len(summary['lattice_rows'])} 个，"
            f"每组 {steps_per_case} 个 KMC step，逐步记录 {summary['step_count']} 条，并行扩展性记录覆盖 1 到 {summary['max_nodes']} 节点。",
            "\\section*{二、测试体系}\n"
            "测试体系采用 Fe-Cu-vacancy 合金体系，包括：\n"
            + tex_enumerate(
                [
                    "Fe 基体；",
                    "Fe-Cu 溶质体系；",
                    "Fe-vacancy 缺陷体系；",
                    "Fe-Cu-vacancy 复合体系；",
                    "Fe-Cu 团簇演化体系。",
                ]
            )
            + "\n\n"
            + latex_escape(
                f"设备统一通过 --device 入口传入，当前记录为 requested={device['requested_device']}，resolved={device['resolved_device']}，backend={device['backend']}，status={device['status']}。"
                "DeepH 调用路径为 DeepHCalculator -> predict_hamiltonian -> total_energy，DeepKS 调用路径为 DeepKSCalculator -> get_potential_energy。"
            ),
            "\\section*{三、分阶段测试内容}",
            tex_table(
                ["阶段", "测试内容", "对应考核指标", "成果形式"],
                [
                    [f"阶段 {r['stage']}", r["test_content"], r["acceptance_indicator"], stage_outcome_text(r["stage"])]
                    for r in summary["stage_rows"]
                ],
                "L{0.11\\linewidth}L{0.36\\linewidth}L{0.27\\linewidth}Y",
            ),
            "\\section*{四、主要输出结果}\n本测试最终形成以下结果：\n"
            + tex_enumerate(
                [
                    "Fe-Cu-vacancy 典型测试算例；",
                    "能量计算结果表；",
                    "软件适配和性能测试记录；",
                    "跨尺度数据集；",
                    "材料演化曲线；",
                    "Cu 团簇组织结构图；",
                    "计算效率对比表；",
                    "材料设计优化建议；",
                    "项目验收报告。",
                ]
            ),
            tex_table(
                ["序号", "主要输出结果", "对应文件"],
                [
                    [1, "Fe-Cu-vacancy 典型测试算例", "outputs/cases/typical_cases.json"],
                    [2, "能量计算结果表", "outputs/tables/energy_results.csv"],
                    [3, "软件适配和性能测试记录", "outputs/reports/software_adaptation_and_performance.md; outputs/tables/performance_records.csv; outputs/tables/module_timing_breakdown.csv"],
                    [4, "跨尺度数据集", "outputs/datasets/multiscale_dataset.csv; outputs/tables/multiscale_dataset.csv"],
                    [5, "材料演化曲线", "outputs/figures/material_evolution_curves.png"],
                    [6, "Cu 团簇组织结构图", "outputs/figures/cu_cluster_structure.png"],
                    [7, "计算效率对比表", "outputs/tables/efficiency_comparison.csv; outputs/figures/runtime_comparison.png"],
                    [8, "材料设计优化建议", "outputs/reports/material_design_recommendations.md"],
                    [9, "项目验收报告", "outputs/reports/acceptance_report.pdf; outputs/reports/acceptance_report.tex; outputs/reports/acceptance_report.md"],
                ],
                "L{0.08\\linewidth}L{0.28\\linewidth}Y",
            ),
            "\\section*{五、验收展示内容}\n验收时重点展示：\n"
            + tex_enumerate(
                [
                    "软件适配结果：说明 Fe-Cu-vacancy 算例可以正常运行；",
                    "性能优化结果：展示优化前后运行时间对比；",
                    "跨尺度数据结果：展示不同温度、成分条件下的数据生成；",
                    "材料演化结果：展示能量变化曲线和 Cu 团簇结构；",
                    "设计优化结果：给出材料成分和组织调控建议；",
                    "验收报告：汇总各阶段任务完成情况。",
                ]
            ),
            "\\subsection*{图形展示}",
            tex_figure("../figures/material_evolution_curves.png", "温度、Cu density 与 vacancy density 条件下的材料演化曲线", "0.94\\linewidth"),
            tex_figure_note("左图展示不同温度条件下平均能量随 KMC step 的变化；中图展示不同 Cu density 条件下最大 Cu 团簇尺寸变化，并去除了最高 Cu density 曲线以突出低中 Cu 含量差异；右图展示不同 vacancy density 条件下物理时间推进差异。"),
            tex_figure("../figures/cu_cluster_structure.png", "Cu 团簇组织结构图", "0.74\\linewidth"),
            tex_figure_note("图中选取最大 Cu 团簇增长最明显的算例，上排给出初始与最终 whole box 的 Cu 原子空间分布，下排给出局部放大区域；红色描边表示当前最大团簇，黄色点表示最终最大团簇中新加入的 Cu 位点，用于突出 initial 到 final 的团簇增长。"),
            tex_figure("../figures/runtime_comparison.png", "优化前后运行时间对比", "0.80\\linewidth"),
            tex_figure_note("左图使用对数坐标比较全量速率刷新 baseline 与增量更新模式的运行时间，右图给出各 lattice size 下的 measured speedup，用于展示性能优化趋势。"),
            "\\section*{六、简要结论}\n"
            "通过 Fe-Cu-vacancy 测试样例，本测试展示了 KMC 从材料能量计算、跨尺度数据生成、大规模并行计算展示到材料设计建议的完整流程。测试结果可支撑强关联材料体系的数值模拟、智能计算和优化设计任务验收。\n\n"
            "关键结果摘要如下：",
            tex_table(
                ["项目", "结果"],
                [
                    ["总 KMC 算例数量", summary["case_count"]],
                    ["逐步 KMC 记录", summary["step_count"]],
                    ["能量结果行数", len(summary["energy_rows"])],
                    ["最终 pair energy 范围", f"{energy_min:.6f} 到 {energy_max:.6f} eV"],
                    ["Cu 最大团簇范围", f"{cluster_min} 到 {cluster_max}"],
                    ["十种 lattice size speedup", speedups],
                    ["DeepH / DeepKS 并行扩展性记录", f"覆盖 1 到 {summary['max_nodes']} 节点"],
                ],
                "L{0.30\\linewidth}Y",
            ),
        ]
    )
    TEX_OUT.write_text(
        tex_document(
            "强关联材料多尺度计算 KMC 测试报告",
            f"Fe-Cu-vacancy 合金体系测试 | {RUN_DATE}",
            body,
            "KMC 测试报告",
        ),
        encoding="utf-8",
    )


def main() -> int:
    summary = make_summary()
    build_markdown(summary)
    build_latex(summary)
    compile_pdf(TEX_OUT)
    refresh_manifest()
    print(PDF_OUT)
    print(TEX_OUT)
    print(MD_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
