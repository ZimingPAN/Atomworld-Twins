# 软件适配和性能测试记录

## 适配范围

- KMC 后端可正常完成 Fe-Cu-vacancy 算例初始化、能量计算、扩散率计算和逐步演化。
- 测试输出包含能量、性能、跨尺度数据、图表和验收报告。
- DeepH / DeepKS 能量接口在初始化阶段输出接口能量与真实调用方式，调用记录已写入 CSV。

## 性能记录摘要

- 8x8x8 baseline_full_recompute: runtime=0.048430s, steps/s=619.449, speedup_vs_baseline=1.000
- 8x8x8 optimized_incremental_rate_update: runtime=0.036555s, steps/s=820.691, speedup_vs_baseline=1.325
- 10x10x10 baseline_full_recompute: runtime=0.063891s, steps/s=469.550, speedup_vs_baseline=1.000
- 10x10x10 optimized_incremental_rate_update: runtime=0.051906s, steps/s=577.965, speedup_vs_baseline=1.231
- 12x12x12 baseline_full_recompute: runtime=0.095191s, steps/s=315.157, speedup_vs_baseline=1.000
- 12x12x12 optimized_incremental_rate_update: runtime=0.082169s, steps/s=365.102, speedup_vs_baseline=1.158

## 主要耗时模块

主要耗时模块按 `rate_recompute_or_refresh`、`action_sampling`、`state_update_energy_reward` 三类记录，详见 `outputs/tables/module_timing_breakdown.csv`。
