# 软件适配和性能测试记录

## 适配范围

- KMC 后端可正常完成 Fe-Cu-vacancy 算例初始化、能量计算、扩散率计算和逐步演化。
- 测试输出包含能量、性能、跨尺度数据、图表和验收报告。
- DeepH / DeepKS 能量接口在初始化阶段输出接口能量与库调用方式，调用记录已写入 CSV。

## 性能记录摘要

- baseline 模式采用 2048 次全量速率刷新，optimized 模式采用增量速率更新。

- 8x8x8 baseline_full_recompute: runtime=42.669550s, steps/s=2.812, speedup_vs_baseline=1.000
- 8x8x8 optimized_incremental_rate_update: runtime=0.152471s, steps/s=787.036, speedup_vs_baseline=279.854
- 10x10x10 baseline_full_recompute: runtime=42.756140s, steps/s=2.807, speedup_vs_baseline=1.000
- 10x10x10 optimized_incremental_rate_update: runtime=0.231333s, steps/s=518.734, speedup_vs_baseline=184.825
- 12x12x12 baseline_full_recompute: runtime=53.884677s, steps/s=2.227, speedup_vs_baseline=1.000
- 12x12x12 optimized_incremental_rate_update: runtime=0.310359s, steps/s=386.649, speedup_vs_baseline=173.620
- 14x14x14 baseline_full_recompute: runtime=52.347620s, steps/s=2.292, speedup_vs_baseline=1.000
- 14x14x14 optimized_incremental_rate_update: runtime=0.624058s, steps/s=192.290, speedup_vs_baseline=83.883
- 16x16x16 baseline_full_recompute: runtime=54.410281s, steps/s=2.205, speedup_vs_baseline=1.000
- 16x16x16 optimized_incremental_rate_update: runtime=0.936572s, steps/s=128.127, speedup_vs_baseline=58.095
- 18x18x18 baseline_full_recompute: runtime=53.313591s, steps/s=2.251, speedup_vs_baseline=1.000
- 18x18x18 optimized_incremental_rate_update: runtime=1.203496s, steps/s=99.709, speedup_vs_baseline=44.299
- 20x20x20 baseline_full_recompute: runtime=60.066228s, steps/s=1.998, speedup_vs_baseline=1.000
- 20x20x20 optimized_incremental_rate_update: runtime=1.769262s, steps/s=67.825, speedup_vs_baseline=33.950
- 22x22x22 baseline_full_recompute: runtime=62.865116s, steps/s=1.909, speedup_vs_baseline=1.000
- 22x22x22 optimized_incremental_rate_update: runtime=2.634929s, steps/s=45.542, speedup_vs_baseline=23.858
- 24x24x24 baseline_full_recompute: runtime=68.801364s, steps/s=1.744, speedup_vs_baseline=1.000
- 24x24x24 optimized_incremental_rate_update: runtime=3.340558s, steps/s=35.922, speedup_vs_baseline=20.596
- 26x26x26 baseline_full_recompute: runtime=115.749780s, steps/s=1.037, speedup_vs_baseline=1.000
- 26x26x26 optimized_incremental_rate_update: runtime=4.680601s, steps/s=25.638, speedup_vs_baseline=24.730

## 主要耗时模块

主要耗时模块按 `rate_recompute_or_refresh`、`action_sampling`、`state_update_energy_reward` 三类记录，详见 `outputs/tables/module_timing_breakdown.csv`。
