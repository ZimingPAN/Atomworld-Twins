# 强关联材料多尺度计算 KMC 测试文档

## 测试概述

本测试围绕 Fe-Cu-vacancy 合金体系，验证 KMC 在材料能量计算、跨尺度数据生成、并行训练展示、组织结构分析和材料设计建议方面的完整流程。

## 关键结果

- 测试体系：Fe 基体、Fe-Cu 溶质体系、Fe-vacancy 缺陷体系、Fe-Cu-vacancy 复合体系、Fe-Cu 团簇演化体系。
- 跨尺度组合：18 个温度、Cu 含量和 vacancy 含量组合。
- 逐步记录：450 条 KMC 演化记录。
- 设备接口：requested=cpu, resolved=cpu, status=active。
- 优化前后 speedup：1.325x, 1.231x, 1.158x。
- 能量范围：-14783.031096 到 -14610.868069 eV。
- Cu 最大团簇范围：1 到 38。
- DeepH / DeepKS 并行训练展示最大节点数：128。

## 材料设计结论

- 单位位点能量最低组合：ms_000，T=300.0 K, Cu=0.005, V=0.001。
- Cu 团簇最大组合：ms_005，T=300.0 K, Cu=0.03, V=0.002, max_cluster=38。

## 主要输出

1. `outputs/cases/typical_cases.json`
2. `outputs/tables/energy_results.csv`
3. `outputs/reports/software_adaptation_and_performance.md`
4. `outputs/datasets/multiscale_dataset.csv`
5. `outputs/figures/material_evolution_curves.png`
6. `outputs/figures/cu_cluster_structure.png`
7. `outputs/tables/efficiency_comparison.csv`
8. `outputs/reports/material_design_recommendations.md`
9. `outputs/reports/acceptance_report.docx`
