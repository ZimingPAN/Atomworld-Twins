#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from kmc_latex_tools import compile_pdf, tex_document, tex_enumerate, tex_figure, tex_itemize, tex_metric_strip, tex_table


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
        f"| {r['stage']} | {r['test_content']} | {r['generated_artifacts']} | {status_label(r['status'])} |"
        for r in summary["stage_rows"]
    )
    text = f"""# 强关联材料多尺度计算 KMC 测试文档

Fe-Cu-vacancy 合金体系测试 | 结果整理版 | {RUN_DATE}

## 0. 测试规模总览

| 项目 | 本次结果 |
| --- | --- |
| 温度扫描 | {temp_values} |
| Cu density 扫描 | {cu_values} |
| vacancy density 扫描 | {v_values} |
| lattice size 扫描 | {lattice_values} |
| 温度/成分/缺陷组合数量 | {len(summary['main_rows'])} |
| lattice size 扫描数量 | {len(summary['lattice_rows'])} |
| 总 KMC 算例数量 | {summary['case_count']} |
| 每组 KMC 步数 | {steps_per_case} |
| 逐步 KMC 记录 | {summary['step_count']} |
| 并行扩展性节点 | 1 到 {summary['max_nodes']} |
| 十种 lattice size speedup | {speedups} |
| 核心数据输出 | `outputs/datasets/multiscale_dataset.csv`; `outputs/tables/multiscale_dataset.csv`; `outputs/tables/energy_results.csv`; `outputs/tables/lattice_size_scan.csv`; `outputs/tables/parallel_training_display.csv` |

## 1. 测试依据与目标

本测试以 Fe-Cu-vacancy 合金体系为样例，围绕材料能量计算、跨尺度数据生成、并行扩展性记录、材料演化分析和材料设计建议形成完整测试闭环。

- 验证 Fe-Cu-vacancy 典型算例可以完成 KMC 初始化、能量计算、扩散率计算和逐步演化。
- 验证 DeepH / DeepKS 能量接口在初始化阶段输出接口能量，并保留库调用路径。
- 验证不同温度、不同 Cu 含量、不同 vacancy 含量和不同 lattice size 条件下的跨尺度数据生成与组织结构分析。
- 验证优化前后运行时间、主要耗时模块、并行扩展性记录和材料设计建议均有可追溯输出。

## 2. 测试体系

| 体系 | 测试目的 | 输出证据 |
| --- | --- | --- |
| Fe 基体 | 建立能量与结构基准 | `outputs/cases/typical_cases.json` |
| Fe-Cu 溶质体系 | 检查 Cu 溶质构型与能量接口 | `outputs/tables/energy_results.csv` |
| Fe-vacancy 缺陷体系 | 检查 vacancy-hop KMC 演化 | `outputs/datasets/multiscale_dataset.csv`; `outputs/tables/multiscale_dataset.csv` |
| Fe-Cu-vacancy 复合体系 | 连接能量、扩散率、性能和跨尺度数据 | `outputs/tables/energy_results.csv`; `outputs/tables/performance_records.csv` |
| Fe-Cu 团簇演化体系 | 展示 Cu 团簇组织结构变化 | `outputs/figures/cu_cluster_structure.png` |

## 3. 测试配置与设备接口

本次运行设备配置为 requested={device['requested_device']}，resolved={device['resolved_device']}，backend={device['backend']}，status={device['status']}。脚本统一通过 `--device` 传入计算设备，并支持 `cpu`、`cuda:localrank` 和 `sdaa:localrank`。

本次跨尺度扫描网格包含温度 {temp_values}，Cu density {cu_values}，vacancy density {v_values}，lattice size {lattice_values}。

| 配置项 | 取值 |
| --- | --- |
| 主执行脚本 | `run_kmc_acceptance.py` |
| 设备入口 | `--device` |
| 严格设备模式 | `--strict-device` |
| DeepH 调用路径 | `DeepHCalculator -> predict_hamiltonian -> total_energy` |
| DeepKS 调用路径 | `DeepKSCalculator -> get_potential_energy` |
| 输出根目录 | `outputs/` |

## 4. 主要输出结果

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

## 5. 验收展示内容

验收时重点展示：

1. 软件适配结果：说明 Fe-Cu-vacancy 算例可以正常运行；
2. 性能优化结果：展示优化前后运行时间对比；
3. 跨尺度数据结果：展示不同温度、成分条件下的数据生成；
4. 材料演化结果：展示能量变化曲线和 Cu 团簇结构；
5. 设计优化结果：给出材料成分和组织调控建议；
6. 验收报告：汇总各阶段任务完成情况。

| 序号 | 验收展示内容 | 展示证据 |
| --- | --- | --- |
| 1 | 软件适配结果：说明 Fe-Cu-vacancy 算例可以正常运行 | `outputs/logs/model_call_log.txt`; `outputs/tables/stage_completion_matrix.csv` |
| 2 | 性能优化结果：展示优化前后运行时间对比 | `outputs/tables/performance_records.csv`; `outputs/figures/runtime_comparison.png` |
| 3 | 跨尺度数据结果：展示不同温度、成分条件下的数据生成 | `outputs/datasets/multiscale_dataset.csv`; `outputs/tables/energy_results.csv` |
| 4 | 材料演化结果：展示能量变化曲线和 Cu 团簇结构 | `outputs/figures/material_evolution_curves.png`; `outputs/figures/cu_cluster_structure.png` |
| 5 | 设计优化结果：给出材料成分和组织调控建议 | `outputs/reports/material_design_recommendations.md`; `outputs/tables/composition_structure_trends.csv` |
| 6 | 验收报告：汇总各阶段任务完成情况 | `outputs/reports/acceptance_report.pdf`; `outputs/reports/output_audit_against_test_plan.md` |

## 6. 分阶段测试完成情况

| 阶段 | 测试内容 | 成果形式 | 状态 |
| --- | --- | --- | --- |
{stage_lines}

## 7. 关键结果

| 结果项 | 数值或结论 |
| --- | --- |
| 温度/成分/缺陷组合数量 | {len(summary['main_rows'])} |
| lattice size 扫描数量 | {len(summary['lattice_rows'])} |
| 总 KMC 算例数量 | {summary['case_count']} |
| 逐步 KMC 记录 | {summary['step_count']} |
| 能量结果行数 | {len(summary['energy_rows'])} |
| 最终 pair energy 范围 | {energy_min:.6f} 到 {energy_max:.6f} eV |
| Cu 最大团簇范围 | {cluster_min} 到 {cluster_max} |
| 十种 lattice size speedup | {speedups} |
| DeepH / DeepKS 并行扩展性记录 | 覆盖 1 到 {summary['max_nodes']} 节点 |
| 输出清单记录文件数 | {summary['manifest_count']}（不含 manifest 自身） |

## 8. 图形结果

- 图 1：`outputs/figures/material_evolution_curves.png`，展示温度下能量变化、Cu density 下团簇演化、vacancy density 下物理时间演化。
- 图 2：`outputs/figures/cu_cluster_structure.png`，展示 Cu 团簇组织结构图。
- 图 3：`outputs/figures/runtime_comparison.png`，展示优化前后运行时间对比。

## 9. 材料设计建议

- 单位位点能量最低组合为 `{best_energy['case_id']}`：T={fmt_temp(best_energy['temperature_K'])}，Cu={best_energy['cu_density']}，V={best_energy['v_density']}。
- Cu 团簇最大组合为 `{best_cluster['case_id']}`：T={fmt_temp(best_cluster['temperature_K'])}，Cu={best_cluster['cu_density']}，V={best_cluster['v_density']}，max_cluster={best_cluster['final_cu_cluster_max']}。
- 若目标是降低能量并保持均匀固溶，建议优先采用 Fe-rich、低到中等 Cu 配方，并控制 vacancy density。
- 若目标是展示 Cu-rich clustering 或析出趋势，建议提高 Cu density，并在中高温条件下延长 KMC 演化步数。

## 10. 输出文件索引与结论

本测试已形成 Fe-Cu-vacancy 体系从典型算例、能量计算、跨尺度演化、性能对比、并行扩展性记录到材料设计建议的完整 KMC 测试链路，输出覆盖方案要求的主要结果，可用于验收汇报和后续复核。

| 文档要求 | 对应输出 |
| --- | --- |
| Fe-Cu-vacancy 典型测试算例 | `outputs/cases/typical_cases.json` |
| 能量计算结果表 | `outputs/tables/energy_results.csv` |
| lattice size 扫描结果表 | `outputs/tables/lattice_size_scan.csv` |
| 软件适配和性能测试记录 | `outputs/reports/software_adaptation_and_performance.md` |
| 跨尺度数据集 | `outputs/datasets/multiscale_dataset.csv`; `outputs/tables/multiscale_dataset.csv` |
| 材料演化曲线 | `outputs/figures/material_evolution_curves.png` |
| Cu 团簇组织结构图 | `outputs/figures/cu_cluster_structure.png` |
| 计算效率对比表 | `outputs/tables/efficiency_comparison.csv` |
| 材料设计优化建议 | `outputs/reports/material_design_recommendations.md` |
| 项目验收报告 | `outputs/reports/acceptance_report.pdf`; `outputs/reports/acceptance_report.tex` |
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text(text, encoding="utf-8")


def build_latex(summary: dict[str, object]) -> None:
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

    body = "\n\n".join(
        [
            tex_metric_strip(
                [
                    ("总 KMC 算例", summary["case_count"]),
                    ("逐步记录", summary["step_count"]),
                    ("lattice size 数", len(summary["lattice_rows"])),
                    ("最大 speedup", f"{max(summary['speedups']):.3f}x"),
                ]
            ),
            "\\section*{0. 测试规模总览}",
            tex_table(
                ["项目", "本次结果"],
                [
                    ["温度扫描", temp_values],
                    ["Cu density 扫描", cu_values],
                    ["vacancy density 扫描", v_values],
                    ["lattice size 扫描", lattice_values],
                    ["温度/成分/缺陷组合数量", len(summary["main_rows"])],
                    ["lattice size 扫描数量", len(summary["lattice_rows"])],
                    ["总 KMC 算例数量", summary["case_count"]],
                    ["每组 KMC 步数", steps_per_case],
                    ["逐步 KMC 记录", summary["step_count"]],
                    ["并行扩展性节点", f"1 到 {summary['max_nodes']}"],
                    ["十种 lattice size speedup", speedups],
                ],
                "L{0.28\\linewidth}Y",
            ),
            "\\section*{1. 测试依据与目标}\n"
            "本测试以 Fe-Cu-vacancy 合金体系为样例，围绕材料能量计算、跨尺度数据生成、并行扩展性记录、材料演化分析和材料设计建议形成完整测试闭环。\n"
            + tex_itemize(
                [
                    "验证 Fe-Cu-vacancy 典型算例可以完成 KMC 初始化、能量计算、扩散率计算和逐步演化。",
                    "验证 DeepH / DeepKS 能量接口在初始化阶段输出接口能量，并保留库调用路径。",
                    "验证不同温度、不同 Cu 含量、不同 vacancy 含量和不同 lattice size 条件下的跨尺度数据生成与组织结构分析。",
                    "验证优化前后运行时间、主要耗时模块、并行扩展性记录和材料设计建议均有可追溯输出。",
                ]
            ),
            "\\section*{2. 测试体系}",
            tex_table(
                ["体系", "测试目的", "输出证据"],
                [
                    ["Fe 基体", "建立能量与结构基准", "outputs/cases/typical_cases.json"],
                    ["Fe-Cu 溶质体系", "检查 Cu 溶质构型与能量接口", "outputs/tables/energy_results.csv"],
                    ["Fe-vacancy 缺陷体系", "检查 vacancy-hop KMC 演化", "outputs/datasets/multiscale_dataset.csv; outputs/tables/multiscale_dataset.csv"],
                    ["Fe-Cu-vacancy 复合体系", "连接能量、扩散率、性能和跨尺度数据", "outputs/tables/energy_results.csv; outputs/tables/performance_records.csv"],
                    ["Fe-Cu 团簇演化体系", "展示 Cu 团簇组织结构变化", "outputs/figures/cu_cluster_structure.png"],
                ],
                "L{0.20\\linewidth}L{0.34\\linewidth}Y",
            ),
            "\\section*{3. 测试配置与设备接口}\n"
            f"本次运行设备配置为 requested={device['requested_device']}，resolved={device['resolved_device']}，backend={device['backend']}，status={device['status']}。"
            "脚本统一通过 --device 传入计算设备，并支持 cpu、cuda:localrank 和 sdaa:localrank。\n\n"
            f"本次跨尺度扫描网格包含温度 {temp_values}，Cu density {cu_values}，vacancy density {v_values}，lattice size {lattice_values}。",
            tex_table(
                ["配置项", "取值"],
                [
                    ["主执行脚本", "run_kmc_acceptance.py"],
                    ["设备入口", "--device"],
                    ["严格设备模式", "--strict-device"],
                    ["DeepH 调用路径", "DeepHCalculator -> predict_hamiltonian -> total_energy"],
                    ["DeepKS 调用路径", "DeepKSCalculator -> get_potential_energy"],
                    ["输出根目录", "outputs/"],
                ],
                "L{0.28\\linewidth}Y",
            ),
            "\\section*{4. 主要输出结果}\n本测试最终形成以下结果：\n"
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
            "\\section*{5. 验收展示内容}\n验收时重点展示：\n"
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
            tex_table(
                ["序号", "验收展示内容", "展示证据"],
                [
                    [1, "软件适配结果：说明 Fe-Cu-vacancy 算例可以正常运行", "outputs/logs/model_call_log.txt; outputs/tables/stage_completion_matrix.csv"],
                    [2, "性能优化结果：展示优化前后运行时间对比", "outputs/tables/performance_records.csv; outputs/figures/runtime_comparison.png"],
                    [3, "跨尺度数据结果：展示不同温度、成分条件下的数据生成", "outputs/datasets/multiscale_dataset.csv; outputs/tables/energy_results.csv"],
                    [4, "材料演化结果：展示能量变化曲线和 Cu 团簇结构", "outputs/figures/material_evolution_curves.png; outputs/figures/cu_cluster_structure.png"],
                    [5, "设计优化结果：给出材料成分和组织调控建议", "outputs/reports/material_design_recommendations.md; outputs/tables/composition_structure_trends.csv"],
                    [6, "验收报告：汇总各阶段任务完成情况", "outputs/reports/acceptance_report.pdf; outputs/reports/output_audit_against_test_plan.md"],
                ],
                "L{0.08\\linewidth}L{0.40\\linewidth}Y",
            ),
            "\\section*{6. 分阶段测试完成情况}",
            tex_table(
                ["阶段", "测试内容", "成果形式", "状态"],
                [
                    [r["stage"], r["test_content"], short_artifacts(r["generated_artifacts"]), status_label(r["status"])]
                    for r in summary["stage_rows"]
                ],
                "L{0.08\\linewidth}L{0.34\\linewidth}Y L{0.10\\linewidth}",
            ),
            "\\section*{7. 关键结果}",
            tex_table(
                ["结果项", "数值或结论"],
                [
                    ["温度/成分/缺陷组合数量", len(summary["main_rows"])],
                    ["lattice size 扫描数量", len(summary["lattice_rows"])],
                    ["总 KMC 算例数量", summary["case_count"]],
                    ["逐步 KMC 记录", summary["step_count"]],
                    ["能量结果行数", len(summary["energy_rows"])],
                    ["最终 pair energy 范围", f"{energy_min:.6f} 到 {energy_max:.6f} eV"],
                    ["Cu 最大团簇范围", f"{cluster_min} 到 {cluster_max}"],
                    ["十种 lattice size speedup", speedups],
                    ["DeepH / DeepKS 并行扩展性记录", f"覆盖 1 到 {summary['max_nodes']} 节点"],
                    ["输出清单记录文件数", f"{summary['manifest_count']}（不含 manifest 自身）"],
                ],
                "L{0.30\\linewidth}Y",
            ),
            "\\clearpage\n\\section*{8. 图形结果}",
            tex_figure("../figures/material_evolution_curves.png", "温度、Cu density 与 vacancy density 条件下的材料演化曲线", "0.94\\linewidth"),
            tex_figure("../figures/cu_cluster_structure.png", "Cu 团簇组织结构图", "0.74\\linewidth"),
            tex_figure("../figures/runtime_comparison.png", "优化前后运行时间对比", "0.80\\linewidth"),
            "\\section*{9. 材料设计建议}",
            tex_itemize(
                [
                    f"单位位点能量最低组合为 {best_energy['case_id']}：T={fmt_temp(best_energy['temperature_K'])}，Cu={best_energy['cu_density']}，V={best_energy['v_density']}。",
                    f"Cu 团簇最大组合为 {best_cluster['case_id']}：T={fmt_temp(best_cluster['temperature_K'])}，Cu={best_cluster['cu_density']}，V={best_cluster['v_density']}，max_cluster={best_cluster['final_cu_cluster_max']}。",
                    "若目标是降低能量并保持均匀固溶，建议优先采用 Fe-rich、低到中等 Cu 配方，并控制 vacancy density。",
                    "若目标是展示 Cu-rich clustering 或析出趋势，建议提高 Cu density，并在中高温条件下延长 KMC 演化步数。",
                ]
            ),
            "\\section*{10. 输出文件索引与结论}\n"
            "本测试已形成 Fe-Cu-vacancy 体系从典型算例、能量计算、跨尺度演化、性能对比、并行扩展性记录到材料设计建议的完整 KMC 测试链路，输出覆盖方案要求的主要结果，可用于验收汇报和后续复核。",
            tex_table(
                ["文档要求", "对应输出"],
                [
                    ["Fe-Cu-vacancy 典型测试算例", "outputs/cases/typical_cases.json"],
                    ["能量计算结果表", "outputs/tables/energy_results.csv"],
                    ["lattice size 扫描结果表", "outputs/tables/lattice_size_scan.csv"],
                    ["软件适配和性能测试记录", "outputs/reports/software_adaptation_and_performance.md"],
                    ["跨尺度数据集", "outputs/datasets/multiscale_dataset.csv; outputs/tables/multiscale_dataset.csv"],
                    ["材料演化曲线", "outputs/figures/material_evolution_curves.png"],
                    ["Cu 团簇组织结构图", "outputs/figures/cu_cluster_structure.png"],
                    ["计算效率对比表", "outputs/tables/efficiency_comparison.csv"],
                    ["材料设计优化建议", "outputs/reports/material_design_recommendations.md"],
                    ["项目验收报告", "outputs/reports/acceptance_report.pdf; outputs/reports/acceptance_report.tex"],
                ],
                "L{0.32\\linewidth}Y",
            ),
        ]
    )
    TEX_OUT.write_text(
        tex_document(
            "强关联材料多尺度计算 KMC 测试文档",
            f"Fe-Cu-vacancy 合金体系测试 | 结果整理版 | {RUN_DATE}",
            body,
            "KMC 测试文档",
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
