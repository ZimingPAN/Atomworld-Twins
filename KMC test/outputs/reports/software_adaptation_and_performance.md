# 软件适配和性能测试记录

## 适配范围

- KMC 后端可正常完成 Fe-Cu-vacancy 算例初始化、能量计算、扩散率计算和逐步演化。
- 测试输出包含能量、性能、跨尺度数据、图表和验收报告。
- DeepH / DeepKS 能量接口在初始化阶段输出接口能量与库调用方式，调用记录已写入 CSV。

## 性能记录摘要

- baseline 模式采用 64 次全量速率刷新，optimized 模式采用增量速率更新。

- 8x8x8 baseline_full_recompute: runtime=1.767181s, steps/s=67.905, speedup_vs_baseline=1.000
- 8x8x8 optimized_incremental_rate_update: runtime=0.149041s, steps/s=805.147, speedup_vs_baseline=11.857
- 10x10x10 baseline_full_recompute: runtime=1.577099s, steps/s=76.089, speedup_vs_baseline=1.000
- 10x10x10 optimized_incremental_rate_update: runtime=0.233074s, steps/s=514.858, speedup_vs_baseline=6.767
- 12x12x12 baseline_full_recompute: runtime=1.683013s, steps/s=71.301, speedup_vs_baseline=1.000
- 12x12x12 optimized_incremental_rate_update: runtime=0.305092s, steps/s=393.325, speedup_vs_baseline=5.516

## 主要耗时模块

主要耗时模块按 `rate_recompute_or_refresh`、`action_sampling`、`state_update_energy_reward` 三类记录，详见 `outputs/tables/module_timing_breakdown.csv`。
