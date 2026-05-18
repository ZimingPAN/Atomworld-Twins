# 强关联材料多尺度计算 KMC 测试验收报告

## 1. 测试目标

本测试围绕 Fe-Cu-vacancy 合金体系，使用 KMC 展示材料能量计算、跨尺度数据生成、性能记录、并行扩展性记录和材料设计建议流程。

## 2. 软件适配结果

- KMC 后端位置：`kmc_backend/RL4KMC/`
- 主执行脚本：`run_kmc_acceptance.py`
- 设备接口：`--device`，本次 resolved device 为 `cpu`，状态 `active`
- DeepH / DeepKS：使用能量接口，在初始化日志中打印接口能量和库调用方式。
- Fe-Cu-vacancy 主算例数量：350
- 跨尺度扫描网格：T=250,300,350,400,500,600,700,800,900,1000 K; Cu=0.0025,0.005,0.01,0.0134,0.02,0.03,0.05; V=0.0005,0.001,0.002,0.003,0.005

## 3. 能量与跨尺度数据

- 能量计算结果表：`outputs/tables/energy_results.csv`
- 跨尺度逐步数据集：`outputs/datasets/multiscale_dataset.csv`
- 快照数据：`outputs/datasets/kmc_snapshots.csv`
- 材料演化曲线：`outputs/figures/material_evolution_curves.png`
- Cu 团簇结构图：`outputs/figures/cu_cluster_structure.png`

## 4. 性能与并行展示

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
- KMC 增量速率更新相对全量重算的最大测得 speedup：11.857x

## 5. 材料设计建议

材料设计优化建议见 `outputs/reports/material_design_recommendations.md`。

## 6. 主要输出结果对照

| 文档要求 | 已生成文件 |
| --- | --- |
| Fe-Cu-vacancy 典型测试算例 | `outputs/cases/typical_cases.json` |
| 能量计算结果表 | `outputs/tables/energy_results.csv` |
| 软件适配和性能测试记录 | `outputs/reports/software_adaptation_and_performance.md`; `outputs/tables/module_timing_breakdown.csv` |
| 跨尺度数据集 | `outputs/datasets/multiscale_dataset.csv` |
| 材料演化曲线 | `outputs/figures/material_evolution_curves.png` |
| Cu 团簇组织结构图 | `outputs/figures/cu_cluster_structure.png` |
| 计算效率对比表 | `outputs/tables/efficiency_comparison.csv` |
| 材料设计优化建议 | `outputs/reports/material_design_recommendations.md` |
| 项目验收报告 | `outputs/reports/acceptance_report.md`; `outputs/reports/acceptance_report.docx` |

## 7. 结论

测试脚本已完成 KMC 横向验收链路：Fe-Cu-vacancy 算例可运行，能量表、跨尺度演化数据、性能对比、千节点并行扩展性记录、组织结构图和设计建议均已生成。DeepH / DeepKS 当前按要求保留能量接口调用入口，可直接绑定对应库的能量计算函数。

## 8. 输出清单

输出清单见 `outputs/manifest.json`，该清单记录除自身外的生成文件。
