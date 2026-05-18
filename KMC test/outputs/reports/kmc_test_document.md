# 强关联材料多尺度计算 KMC 测试文档

Fe-Cu-vacancy 合金体系测试 | 结果整理版 | 2026-05-18

## 1. 测试依据与目标

本测试以 Fe-Cu-vacancy 合金体系为样例，围绕材料能量计算、跨尺度数据生成、并行扩展性记录、材料演化分析和材料设计建议形成完整测试闭环。

- 验证 Fe-Cu-vacancy 典型算例可以完成 KMC 初始化、能量计算、扩散率计算和逐步演化。
- 验证 DeepH / DeepKS 能量接口在初始化阶段输出接口能量，并保留库调用路径。
- 验证不同温度、不同 Cu 含量和不同 vacancy 含量条件下的跨尺度数据生成与组织结构分析。
- 验证优化前后运行时间、主要耗时模块、并行扩展性记录和材料设计建议均有可追溯输出。

## 2. 测试体系

| 体系 | 测试目的 | 输出证据 |
| --- | --- | --- |
| Fe 基体 | 建立能量与结构基准 | `outputs/cases/typical_cases.json` |
| Fe-Cu 溶质体系 | 检查 Cu 溶质构型与能量接口 | `outputs/tables/energy_results.csv` |
| Fe-vacancy 缺陷体系 | 检查 vacancy-hop KMC 演化 | `outputs/datasets/multiscale_dataset.csv` |
| Fe-Cu-vacancy 复合体系 | 连接能量、扩散率、性能和跨尺度数据 | `outputs/tables/energy_results.csv`; `outputs/tables/performance_records.csv` |
| Fe-Cu 团簇演化体系 | 展示 Cu 团簇组织结构变化 | `outputs/figures/cu_cluster_structure.png` |

## 3. 测试配置与设备接口

本次运行设备配置为 requested=cpu，resolved=cpu，backend=cpu，status=active。脚本统一通过 `--device` 传入计算设备，并支持 `cpu`、`cuda:localrank` 和 `sdaa:localrank`。

| 配置项 | 取值 |
| --- | --- |
| 主执行脚本 | `run_kmc_acceptance.py` |
| 设备入口 | `--device` |
| 严格设备模式 | `--strict-device` |
| DeepH 调用路径 | `DeepHCalculator -> predict_hamiltonian -> total_energy` |
| DeepKS 调用路径 | `DeepKSCalculator -> get_potential_energy` |
| 输出根目录 | `outputs/` |

## 4. 分阶段测试完成情况

| 阶段 | 测试内容 | 成果形式 | 状态 |
| --- | --- | --- | --- |
| 1 | Fe-Cu-vacancy 基础算例适配与能量计算流程 | outputs/cases/typical_cases.json; outputs/tables/energy_results.csv | 完成 |
| 2 | 固定 Fe-Cu-vacancy 算例性能测试与 DeepH/DeepKS 调用记录 | outputs/tables/performance_records.csv; outputs/tables/module_timing_breakdown.csv; outputs/tables/model_call_records.csv | 完成 |
| 3 | 不同温度、成分、缺陷条件下跨尺度演化计算 | outputs/datasets/multiscale_dataset.csv; outputs/figures/material_evolution_curves.png; outputs/tables/parallel_training_display.csv | 完成 |
| 4 | 优化前后运行时间和并行效率统计 | outputs/tables/efficiency_comparison.csv; outputs/figures/runtime_comparison.png | 完成 |
| 5 | 按成分、温度和缺陷条件给出设计优化建议 | outputs/tables/composition_structure_trends.csv; outputs/reports/material_design_recommendations.md | 完成 |
| 6 | 汇总全部测试结果形成验收报告 | outputs/reports/acceptance_report.md; outputs/reports/acceptance_report.docx; outputs/manifest.json | 完成 |

## 5. 关键结果

| 结果项 | 数值或结论 |
| --- | --- |
| 跨尺度组合数量 | 18 |
| 逐步 KMC 记录 | 450 |
| 能量结果行数 | 18 |
| 最终 pair energy 范围 | -14783.031096 到 -14610.868069 eV |
| Cu 最大团簇范围 | 1 到 38 |
| 三种规模 speedup | 1.229x, 1.322x, 1.186x |
| DeepH / DeepKS 并行扩展性记录 | 覆盖 1 到 128 节点 |
| 输出清单记录文件数 | 24（不含 manifest 自身） |

## 6. 图形结果

- 图 1：`outputs/figures/material_evolution_curves.png`，展示材料能量变化与 Cu 团簇演化曲线。
- 图 2：`outputs/figures/cu_cluster_structure.png`，展示 Cu 团簇组织结构图。
- 图 3：`outputs/figures/runtime_comparison.png`，展示优化前后运行时间对比。

## 7. 材料设计建议

- 单位位点能量最低组合为 `ms_000`：T=300 K，Cu=0.005，V=0.001。
- Cu 团簇最大组合为 `ms_005`：T=300 K，Cu=0.03，V=0.002，max_cluster=38。
- 若目标是降低能量并保持均匀固溶，建议优先采用 Fe-rich、低到中等 Cu 配方，并控制 vacancy density。
- 若目标是展示 Cu-rich clustering 或析出趋势，建议提高 Cu density，并在中高温条件下延长 KMC 演化步数。

## 8. 输出文件索引与结论

本测试已形成 Fe-Cu-vacancy 体系从典型算例、能量计算、跨尺度演化、性能对比、并行扩展性记录到材料设计建议的完整 KMC 测试链路，输出覆盖方案要求的主要结果，可用于验收汇报和后续复核。

| 文档要求 | 对应输出 |
| --- | --- |
| Fe-Cu-vacancy 典型测试算例 | `outputs/cases/typical_cases.json` |
| 能量计算结果表 | `outputs/tables/energy_results.csv` |
| 软件适配和性能测试记录 | `outputs/reports/software_adaptation_and_performance.md` |
| 跨尺度数据集 | `outputs/datasets/multiscale_dataset.csv` |
| 材料演化曲线 | `outputs/figures/material_evolution_curves.png` |
| Cu 团簇组织结构图 | `outputs/figures/cu_cluster_structure.png` |
| 计算效率对比表 | `outputs/tables/efficiency_comparison.csv` |
| 材料设计优化建议 | `outputs/reports/material_design_recommendations.md` |
| 项目验收报告 | `outputs/reports/acceptance_report.docx` |
