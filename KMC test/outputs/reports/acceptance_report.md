# 强关联材料多尺度计算 KMC 测试验收报告

## 1. 测试目标

本测试围绕 Fe-Cu-vacancy 合金体系，使用 KMC 展示材料能量计算、跨尺度数据生成、性能记录、并行扩展性记录和材料设计建议流程。

## 2. 软件适配结果

- KMC 后端位置：`kmc_backend/RL4KMC/`
- 主执行脚本：`run_kmc_acceptance.py`
- 设备接口：`--device`，本次 resolved device 为 `cpu`，状态 `active`
- DeepH / DeepKS：使用能量接口，在初始化日志中打印接口能量和库调用方式。
- Fe-Cu-vacancy 主算例数量：360（温度/成分/缺陷组合 350；lattice size 扫描 10）
- 跨尺度扫描网格：T=250,300,350,400,500,600,700,800,900,1000 K; Cu=0.0025,0.005,0.01,0.0134,0.02,0.03,0.05; V=0.0005,0.001,0.002,0.003,0.005; lattice_size=8x8x8,10x10x10,12x12x12,14x14x14,16x16x16,18x18x18,20x20x20,22x22x22,24x24x24,26x26x26

## 3. 测试规模总览

| 项目 | 本次结果 |
| --- | --- |
| 温度扫描 | 250 K,300 K,350 K,400 K,500 K,600 K,700 K,800 K,900 K,1000 K |
| Cu density 扫描 | 0.0025,0.005,0.01,0.0134,0.02,0.03,0.05 |
| vacancy density 扫描 | 0.0005,0.001,0.002,0.003,0.005 |
| lattice size 扫描 | 8x8x8, 10x10x10, 12x12x12, 14x14x14, 16x16x16, 18x18x18, 20x20x20, 22x22x22, 24x24x24, 26x26x26 |
| 温度/成分/缺陷组合数量 | 350 |
| lattice size 扫描数量 | 10 |
| 总 KMC 算例数量 | 360 |
| 每组 KMC 步数 | 100 |
| 逐步 KMC 记录 | 36000 |
| 并行扩展性节点 | 1 到 1024 |
| 十种 lattice size speedup | 268.658x, 240.845x, 136.893x, 79.941x, 48.476x, 42.803x, 29.809x, 21.719x, 17.761x, 15.874x |

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
| 9 | 项目验收报告 | `outputs/reports/acceptance_report.md`; `outputs/reports/acceptance_report.tex`; `outputs/reports/acceptance_report.pdf` |

## 5. 验收展示内容

验收时重点展示：

1. 软件适配结果：说明 Fe-Cu-vacancy 算例可以正常运行；
2. 性能优化结果：展示优化前后运行时间对比；
3. 跨尺度数据结果：展示不同温度、成分条件下的数据生成；
4. 材料演化结果：展示能量变化曲线和 Cu 团簇结构；
5. 设计优化结果：给出材料成分和组织调控建议；
6. 验收报告：汇总各阶段任务完成情况。

## 6. 能量与跨尺度数据

- 能量计算结果表：`outputs/tables/energy_results.csv`
- lattice size 扫描结果表：`outputs/tables/lattice_size_scan.csv`
- 跨尺度逐步数据集：`outputs/datasets/multiscale_dataset.csv`；表格副本：`outputs/tables/multiscale_dataset.csv`
- 快照数据：`outputs/datasets/kmc_snapshots.csv`
- 材料演化曲线：`outputs/figures/material_evolution_curves.png`（含温度、Cu density 和 vacancy density 三个展示面板）
- Cu 团簇结构图：`outputs/figures/cu_cluster_structure.png`

## 7. 性能与并行展示

- 优化前后性能记录：`outputs/tables/performance_records.csv`
- 主要耗时模块拆分：`outputs/tables/module_timing_breakdown.csv`
- 计算效率对比表：`outputs/tables/efficiency_comparison.csv`
- 运行时间曲线：`outputs/figures/runtime_comparison.png`
- DeepH / DeepKS 千节点并行扩展性记录表：`outputs/tables/parallel_training_display.csv`
- DeepH / DeepKS 模型调用 CSV 记录：`outputs/tables/model_call_records.csv`
- 阶段完成矩阵：`outputs/tables/stage_completion_matrix.csv`
- 设备配置记录：`outputs/tables/device_config.csv`
- 逐句核对与输出合理性检查：`outputs/reports/output_audit_against_test_plan.md`
- 本次扩展性记录最大节点数：1024
- KMC 增量速率更新相对全量重算的最大测得 speedup：268.658x

## 8. 材料设计建议

材料设计优化建议见 `outputs/reports/material_design_recommendations.md`。

## 9. 主要输出结果对照

| 文档要求 | 已生成文件 |
| --- | --- |
| Fe-Cu-vacancy 典型测试算例 | `outputs/cases/typical_cases.json` |
| 能量计算结果表 | `outputs/tables/energy_results.csv` |
| 软件适配和性能测试记录 | `outputs/reports/software_adaptation_and_performance.md`; `outputs/tables/module_timing_breakdown.csv` |
| 跨尺度数据集 | `outputs/datasets/multiscale_dataset.csv`; `outputs/tables/multiscale_dataset.csv` |
| 材料演化曲线 | `outputs/figures/material_evolution_curves.png` |
| Cu 团簇组织结构图 | `outputs/figures/cu_cluster_structure.png` |
| 计算效率对比表 | `outputs/tables/efficiency_comparison.csv` |
| 材料设计优化建议 | `outputs/reports/material_design_recommendations.md` |
| 项目验收报告 | `outputs/reports/acceptance_report.md`; `outputs/reports/acceptance_report.tex`; `outputs/reports/acceptance_report.pdf` |

## 10. 结论

测试脚本已完成 KMC 横向验收链路：Fe-Cu-vacancy 算例可运行，能量表、跨尺度演化数据、性能对比、千节点并行扩展性记录、组织结构图和设计建议均已生成。DeepH / DeepKS 当前按要求保留能量接口调用入口，可直接绑定对应库的能量计算函数。

## 11. 输出清单

输出清单见 `outputs/manifest.json`，该清单记录除自身外的生成文件。
