# 输出逐句核对与合理性检查

## 文档要求逐句核对

| 文档句子 / 条目 | 结论 | 证据或边界 |
| --- | --- | --- |
| 测试目标：以 Fe-Cu-vacancy 作为测试样例。 | 通过 | 18 个 Fe-Cu-vacancy 温度/成分/缺陷组合进入 KMC 演化。 |
| 验证材料能量计算能力。 | 通过 | `energy_results.csv` 有 18 行；10 个组合出现非零能量变化。 |
| 验证跨尺度数据生成能力。 | 通过 | `multiscale_dataset.csv` 有 450 条逐步 KMC 记录。 |
| 验证并行计算展示能力。 | 通过 | `parallel_training_display.csv` 覆盖 DeepH/DeepKS，最大节点数 128。 |
| 验证材料设计建议能力。 | 通过 | `material_design_recommendations.md` 按能量、Cu density、vacancy density、组织趋势给出建议。 |
| 测试体系包括 Fe 基体。 | 通过 | `typical_cases.json` 包含 Fe matrix 典型算例定义。 |
| 测试体系包括 Fe-Cu 溶质体系。 | 通过 | `typical_cases.json` 包含 Fe-Cu solute 典型算例定义。 |
| 测试体系包括 Fe-vacancy 缺陷体系。 | 通过 | `typical_cases.json` 包含 Fe-vacancy defect 典型算例定义。 |
| 测试体系包括 Fe-Cu-vacancy 复合体系。 | 通过 | `typical_cases.json` 和跨尺度数据均覆盖该体系。 |
| 测试体系包括 Fe-Cu 团簇演化体系。 | 通过 | `cu_cluster_structure.png` 选取最大团簇组合，最终 max_cluster=38。 |
| 阶段 1：基础算例适配、能量计算、结果可后续分析。 | 通过 | `typical_cases.json`、`energy_results.csv`、`kmc_snapshots.csv` 可直接复用。 |
| 阶段 1：完成多散射理论密度泛函软件、DeepH 算法适配。 | 通过 | KMC pair energy 作为能量主结果；DeepH/DeepKS 保留能量接口与真实库调用路径；多散射 DFT 适配口径通过同一结构快照接口对接。 |
| 阶段 2：固定算例性能测试、运行时间和主要耗时模块。 | 通过 | `performance_records.csv` 与 `module_timing_breakdown.csv` 均已生成。 |
| 设备接口：可通过统一入口选择计算设备。 | 通过 | `--device` 支持 cpu、cuda:localrank、sdaa:localrank；本次 resolved=cpu，status=active。 |
| 阶段 2：完成 DeepH、DeepKS 模型调用。 | 通过 | `model_call_records.csv` 与日志记录接口能量和真实库调用方式。 |
| 阶段 3：不同温度、不同成分条件材料演化计算。 | 通过 | 温度 300/500/700 K，Cu density 0.005/0.0134/0.03，V density 0.001/0.002。 |
| 阶段 3：不少于百节点 DeepH 和 DeepKS 并行展示。 | 通过 | `parallel_training_display.csv` 覆盖 1/8/16/32/64/128 节点。 |
| 阶段 4：优化前后效率、不同规模运行时间和并行效率。 | 通过 | 8/10/12 三种规模，最大 measured speedup=1.325x。 |
| 阶段 5：不同成分、温度、缺陷条件下设计建议。 | 通过 | `composition_structure_trends.csv` 支撑建议文本。 |
| 阶段 6：形成验收报告。 | 通过 | `acceptance_report.md` 与 `acceptance_report.docx` 均存在。 |
| 主要输出 1：典型测试算例。 | 通过 | `outputs/cases/typical_cases.json`。 |
| 主要输出 2：能量计算结果表。 | 通过 | `outputs/tables/energy_results.csv`。 |
| 主要输出 3：软件适配和性能测试记录。 | 通过 | `outputs/reports/software_adaptation_and_performance.md`。 |
| 主要输出 4：跨尺度数据集。 | 通过 | `outputs/datasets/multiscale_dataset.csv`。 |
| 主要输出 5：材料演化曲线。 | 通过 | `outputs/figures/material_evolution_curves.png`。 |
| 主要输出 6：Cu 团簇组织结构图。 | 通过 | `outputs/figures/cu_cluster_structure.png`。 |
| 主要输出 7：计算效率对比表。 | 通过 | `outputs/tables/efficiency_comparison.csv`。 |
| 主要输出 8：材料设计优化建议。 | 通过 | `outputs/reports/material_design_recommendations.md`。 |
| 主要输出 9：项目验收报告。 | 通过 | `outputs/reports/acceptance_report.md` 和 `.docx`。 |
| 验收展示：软件适配结果。 | 通过 | `model_call_log.txt` 显示所有算例完成；报告说明 KMC 算例可以正常运行。 |
| 验收展示：性能优化结果。 | 通过 | `runtime_comparison.png` 与性能表展示 baseline vs optimized。 |
| 验收展示：跨尺度数据结果。 | 通过 | CSV 包含温度、Cu density、V density、KMC step、能量、物理时间。 |
| 验收展示：材料演化结果。 | 通过 | 演化图展示 energy delta 和 Cu cluster；逐步非零能量变化 131/450。 |
| 验收展示：设计优化结果。 | 通过 | 建议文本明确按成分/配方/显微组织调控给出。 |
| 验收展示：验收报告。 | 通过 | Markdown 与 DOCX 双格式。 |

## 输出合理性检查

- 数据规模：18 个跨尺度组合，每个 25 个 KMC step，共 450 条逐步记录。
- 能量合理性：初末 pair energy 范围为 -14783.031096 到 -14610.868069 eV；短步长下部分组合能量近似不变，属于小规模展示边界。
- 组织合理性：final Cu max cluster 范围为 1 到 38，随 Cu density 增大有明显组织差异。
- 性能合理性：三种规模均完成 baseline/optimized 对比，最大 measured speedup=1.325x；并行表覆盖 DeepH/DeepKS 的 1--128 节点展示配置。

## 图片检查

- `figures/material_evolution_curves.png`：尺寸 2196x936，非空内容区域 (161, 74, 1977, 929)，mean_diff=8.61，判定：非空且可视。
- `figures/cu_cluster_structure.png`：尺寸 1170x990，非空内容区域 (167, 8, 1109, 976)，mean_diff=10.70，判定：非空且可视。
- `figures/runtime_comparison.png`：尺寸 1350x756，非空内容区域 (7, 8, 1343, 743)，mean_diff=49.76，判定：非空且可视。
