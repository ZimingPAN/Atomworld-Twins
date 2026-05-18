# 强关联材料多尺度计算 KMC 测试文档

Fe-Cu-vacancy 合金体系测试 | 结果整理版 | 2026-05-18

## 0. 测试规模总览

| 项目 | 本次结果 |
| --- | --- |
| 温度扫描 | 250 K, 300 K, 350 K, 400 K, 500 K, 600 K, 700 K, 800 K, 900 K, 1000 K |
| Cu density 扫描 | 0.0025, 0.005, 0.01, 0.0134, 0.02, 0.03, 0.05 |
| vacancy density 扫描 | 0.0005, 0.001, 0.002, 0.003, 0.005 |
| lattice size 扫描 | 8x8x8, 10x10x10, 12x12x12, 14x14x14, 16x16x16, 18x18x18, 20x20x20, 22x22x22, 24x24x24, 26x26x26 |
| 温度/成分/缺陷组合数量 | 350 |
| lattice size 扫描数量 | 10 |
| 总 KMC 算例数量 | 360 |
| 每组 KMC 步数 | 100 |
| 逐步 KMC 记录 | 36000 |
| 并行扩展性节点 | 1 到 1024 |
| 十种 lattice size speedup | 268.658x, 240.845x, 136.893x, 79.941x, 48.476x, 42.803x, 29.809x, 21.719x, 17.761x, 15.874x |
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

本次运行设备配置为 requested=cpu，resolved=cpu，backend=cpu，status=active。脚本统一通过 `--device` 传入计算设备，并支持 `cpu`、`cuda:localrank` 和 `sdaa:localrank`。

本次跨尺度扫描网格包含温度 250 K, 300 K, 350 K, 400 K, 500 K, 600 K, 700 K, 800 K, 900 K, 1000 K，Cu density 0.0025, 0.005, 0.01, 0.0134, 0.02, 0.03, 0.05，vacancy density 0.0005, 0.001, 0.002, 0.003, 0.005，lattice size 8x8x8, 10x10x10, 12x12x12, 14x14x14, 16x16x16, 18x18x18, 20x20x20, 22x22x22, 24x24x24, 26x26x26。

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
| 1 | Fe-Cu-vacancy 基础算例适配与能量计算流程 | outputs/cases/typical_cases.json; outputs/tables/energy_results.csv | 完成 |
| 2 | 固定 Fe-Cu-vacancy 算例性能测试与 DeepH/DeepKS 调用记录 | outputs/tables/performance_records.csv; outputs/tables/module_timing_breakdown.csv; outputs/tables/model_call_records.csv | 完成 |
| 3 | 不同温度、成分、缺陷、lattice size 条件下跨尺度演化计算 | outputs/datasets/multiscale_dataset.csv; outputs/tables/multiscale_dataset.csv; outputs/tables/lattice_size_scan.csv; outputs/figures/material_evolution_curves.png; outputs/tables/parallel_training_display.csv | 完成 |
| 4 | 优化前后运行时间和并行效率统计 | outputs/tables/efficiency_comparison.csv; outputs/figures/runtime_comparison.png | 完成 |
| 5 | 按成分、温度和缺陷条件给出设计优化建议 | outputs/tables/composition_structure_trends.csv; outputs/reports/material_design_recommendations.md | 完成 |
| 6 | 汇总全部测试结果形成验收报告 | outputs/reports/acceptance_report.md; outputs/reports/acceptance_report.tex; outputs/reports/acceptance_report.pdf; outputs/manifest.json | 完成 |

## 7. 关键结果

| 结果项 | 数值或结论 |
| --- | --- |
| 温度/成分/缺陷组合数量 | 350 |
| lattice size 扫描数量 | 10 |
| 总 KMC 算例数量 | 360 |
| 逐步 KMC 记录 | 36000 |
| 能量结果行数 | 360 |
| 最终 pair energy 范围 | -149825.616261 到 -4364.526262 eV |
| Cu 最大团簇范围 | 1 到 167 |
| 十种 lattice size speedup | 268.658x, 240.845x, 136.893x, 79.941x, 48.476x, 42.803x, 29.809x, 21.719x, 17.761x, 15.874x |
| DeepH / DeepKS 并行扩展性记录 | 覆盖 1 到 1024 节点 |
| 输出清单记录文件数 | 32（不含 manifest 自身） |

## 8. 图形结果

- 图 1：`outputs/figures/material_evolution_curves.png`，展示温度下能量变化、Cu density 下团簇演化、vacancy density 下物理时间演化。
  - 图示说明：左图展示不同温度条件下平均能量随 KMC step 的变化；中图展示不同 Cu density 条件下最大 Cu 团簇尺寸变化，并去除了最高 Cu density 曲线以突出低中 Cu 含量差异；右图展示不同 vacancy density 条件下物理时间推进差异。
- 图 2：`outputs/figures/cu_cluster_structure.png`，展示 Cu 团簇组织结构图。
  - 图示说明：图中选取最大 Cu 团簇增长最明显的算例，上排给出初始与最终 whole box 的 Cu 原子空间分布，下排给出局部放大区域；红色描边表示当前最大团簇，黄色点表示最终最大团簇中新加入的 Cu 位点，用于突出 initial 到 final 的团簇增长。
- 图 3：`outputs/figures/runtime_comparison.png`，展示优化前后运行时间对比。
  - 图示说明：左图使用对数坐标比较全量速率刷新 baseline 与增量更新模式的运行时间，右图给出各 lattice size 下的 measured speedup，用于展示性能优化趋势。

## 9. 材料设计建议

- 单位位点能量最低组合为 `ms_315`：T=1000 K，Cu=0.0025，V=0.0005。
- Cu 团簇最大组合为 `ms_034`：T=250 K，Cu=0.05，V=0.005，max_cluster=167。

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
