# 软件适配和性能测试记录

## 适配范围

- KMC 后端可正常完成 Fe-Cu-vacancy 算例初始化、能量计算、扩散率计算和逐步演化。
- 测试输出包含能量、性能、跨尺度数据、图表和验收报告。
- DeepH / DeepKS 能量接口在初始化阶段输出接口能量与库调用方式，调用记录已写入 CSV。

## 性能记录摘要

- 8x8x8 baseline_full_recompute: runtime=0.053677s, steps/s=558.902, speedup_vs_baseline=1.000
- 8x8x8 optimized_incremental_rate_update: runtime=0.043682s, steps/s=686.787, speedup_vs_baseline=1.229
- 10x10x10 baseline_full_recompute: runtime=0.068424s, steps/s=438.442, speedup_vs_baseline=1.000
- 10x10x10 optimized_incremental_rate_update: runtime=0.051752s, steps/s=579.688, speedup_vs_baseline=1.322
- 12x12x12 baseline_full_recompute: runtime=0.095282s, steps/s=314.856, speedup_vs_baseline=1.000
- 12x12x12 optimized_incremental_rate_update: runtime=0.080315s, steps/s=373.530, speedup_vs_baseline=1.186

## 主要耗时模块

主要耗时模块按 `rate_recompute_or_refresh`、`action_sampling`、`state_update_energy_reward` 三类记录，详见 `outputs/tables/module_timing_breakdown.csv`。
