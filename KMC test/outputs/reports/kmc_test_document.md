# 强关联材料多尺度计算 KMC 测试报告

Fe-Cu-vacancy 合金体系测试 | 2026-05-18

## 一、测试目标

围绕强关联材料多尺度计算测试任务，选取 Fe-Cu-vacancy 合金体系作为测试样例，验证 KMC 流程在材料能量计算、跨尺度数据生成、并行计算展示和材料设计建议方面的能力。

本次测试覆盖温度 250 K, 300 K, 350 K, 400 K, 500 K, 600 K, 700 K, 800 K, 900 K, 1000 K，Cu density 0.0025, 0.005, 0.01, 0.0134, 0.02, 0.03, 0.05，vacancy density 0.0005, 0.001, 0.002, 0.003, 0.005，lattice size 8x8x8, 10x10x10, 12x12x12, 14x14x14, 16x16x16, 18x18x18, 20x20x20, 22x22x22, 24x24x24, 26x26x26。共形成 360 个 KMC 算例，其中温度/成分/缺陷组合 350 个，lattice size 扫描 10 个，每组 100 个 KMC step，逐步记录 36000 条，并行扩展性记录覆盖 1 到 1024 节点。

## 二、测试体系

测试体系采用 Fe-Cu-vacancy 合金体系，包括：

1. Fe 基体；
2. Fe-Cu 溶质体系；
3. Fe-vacancy 缺陷体系；
4. Fe-Cu-vacancy 复合体系；
5. Fe-Cu 团簇演化体系。

设备统一通过 `--device` 入口传入，当前记录为 requested=cpu，resolved=cpu，backend=cpu，status=active。DeepH 调用路径为 `DeepHCalculator -> predict_hamiltonian -> total_energy`，DeepKS 调用路径为 `DeepKSCalculator -> get_potential_energy`。

## 三、分阶段测试内容

| 阶段 | 测试内容 | 对应考核指标 | 成果形式 |
| --- | --- | --- | --- |
| 阶段 1 | Fe-Cu-vacancy 基础算例适配与能量计算流程 | 完成多散射/DeepH 能量接口适配 | 测试算例、能量结果 |
| 阶段 2 | 固定 Fe-Cu-vacancy 算例性能测试与 DeepH/DeepKS 调用记录 | 性能对比表、模型调用记录 | 性能对比表、模型调用记录 |
| 阶段 3 | 不同温度、成分、缺陷、lattice size 条件下跨尺度演化计算 | 跨尺度数据集、lattice size 扫描、演化曲线、千节点并行扩展性记录 | 跨尺度数据集、演化曲线 |
| 阶段 4 | 优化前后运行时间和并行效率统计 | 效率对比表、运行时间曲线 | 效率对比表、运行时间曲线 |
| 阶段 5 | 按成分、温度和缺陷条件给出设计优化建议 | 成分-组织趋势表、材料设计建议 | 成分-组织趋势表、材料设计建议 |
| 阶段 6 | 汇总全部测试结果形成验收报告 | 验收报告、测试数据、结果图表 | 验收报告、测试数据、结果图表 |

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
| 总 KMC 算例数量 | 360 |
| 逐步 KMC 记录 | 36000 |
| 能量结果行数 | 360 |
| 最终 pair energy 范围 | -149825.616261 到 -4364.526262 eV |
| Cu 最大团簇范围 | 1 到 167 |
| 十种 lattice size speedup | 268.658x, 240.845x, 136.893x, 79.941x, 48.476x, 42.803x, 29.809x, 21.719x, 17.761x, 15.874x |
| DeepH / DeepKS 并行扩展性记录 | 覆盖 1 到 1024 节点 |
