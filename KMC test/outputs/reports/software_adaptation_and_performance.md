# 软件适配和性能测试记录

## 适配范围

- KMC 后端可正常完成 Fe-Cu-vacancy 算例初始化、能量计算、扩散率计算和逐步演化。
- 测试输出包含能量、性能、跨尺度数据、图表和验收报告。
- DeepH / DeepKS 能量接口在初始化阶段输出接口能量与库调用方式，调用记录已写入 CSV。

## 性能记录摘要

- baseline 模式采用 2048 次全量速率刷新，optimized 模式采用增量速率更新。

- 8x8x8 baseline_full_recompute: runtime=35.476192s, steps/s=2.819, speedup_vs_baseline=1.000
- 8x8x8 optimized_incremental_rate_update: runtime=0.132050s, steps/s=757.290, speedup_vs_baseline=268.658
- 10x10x10 baseline_full_recompute: runtime=46.616472s, steps/s=2.145, speedup_vs_baseline=1.000
- 10x10x10 optimized_incremental_rate_update: runtime=0.193554s, steps/s=516.651, speedup_vs_baseline=240.845
- 12x12x12 baseline_full_recompute: runtime=36.677812s, steps/s=2.726, speedup_vs_baseline=1.000
- 12x12x12 optimized_incremental_rate_update: runtime=0.267931s, steps/s=373.230, speedup_vs_baseline=136.893
- 14x14x14 baseline_full_recompute: runtime=37.503147s, steps/s=2.666, speedup_vs_baseline=1.000
- 14x14x14 optimized_incremental_rate_update: runtime=0.469133s, steps/s=213.159, speedup_vs_baseline=79.941
- 16x16x16 baseline_full_recompute: runtime=41.330529s, steps/s=2.420, speedup_vs_baseline=1.000
- 16x16x16 optimized_incremental_rate_update: runtime=0.852600s, steps/s=117.288, speedup_vs_baseline=48.476
- 18x18x18 baseline_full_recompute: runtime=41.901815s, steps/s=2.387, speedup_vs_baseline=1.000
- 18x18x18 optimized_incremental_rate_update: runtime=0.978949s, steps/s=102.150, speedup_vs_baseline=42.803
- 20x20x20 baseline_full_recompute: runtime=43.168332s, steps/s=2.317, speedup_vs_baseline=1.000
- 20x20x20 optimized_incremental_rate_update: runtime=1.448147s, steps/s=69.054, speedup_vs_baseline=29.809
- 22x22x22 baseline_full_recompute: runtime=46.376671s, steps/s=2.156, speedup_vs_baseline=1.000
- 22x22x22 optimized_incremental_rate_update: runtime=2.135285s, steps/s=46.832, speedup_vs_baseline=21.719
- 24x24x24 baseline_full_recompute: runtime=49.493243s, steps/s=2.020, speedup_vs_baseline=1.000
- 24x24x24 optimized_incremental_rate_update: runtime=2.786578s, steps/s=35.886, speedup_vs_baseline=17.761
- 26x26x26 baseline_full_recompute: runtime=64.008389s, steps/s=1.562, speedup_vs_baseline=1.000
- 26x26x26 optimized_incremental_rate_update: runtime=4.032368s, steps/s=24.799, speedup_vs_baseline=15.874

## 主要耗时模块

主要耗时模块按 `rate_recompute_or_refresh`、`action_sampling`、`state_update_energy_reward` 三类记录，详见 `outputs/tables/module_timing_breakdown.csv`。
