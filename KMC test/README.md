# KMC Acceptance Test

This folder contains an isolated KMC acceptance workflow, generated results, and the report files needed for review.

## Run

```bash
python run_kmc_acceptance.py
```

All generated files are written under `outputs/`. The script recreates that folder on each run.

Default workload:

- Temperature grid: `250, 300, 350, 400, 500, 600, 700, 800, 900, 1000 K`
- Cu density grid: `0.0025, 0.005, 0.01, 0.0134, 0.02, 0.03, 0.05`
- Vacancy density grid: `0.0005, 0.001, 0.002, 0.003, 0.005`
- Lattice size scan: `8x8x8, 10x10x10, 12x12x12, 14x14x14, 16x16x16, 18x18x18, 20x20x20, 22x22x22, 24x24x24, 26x26x26`
- Temperature/composition/defect cases: `350`
- Lattice size cases: `10`
- Total KMC cases: `360`
- KMC steps per case: `100`
- Parallel scaling record: up to `1024` nodes

Device selection uses one interface:

```bash
python run_kmc_acceptance.py --device cpu
python run_kmc_acceptance.py --device cuda:localrank
python run_kmc_acceptance.py --device sdaa:localrank
python run_kmc_acceptance.py --device cuda:0 --local-rank 0
python run_kmc_acceptance.py --device cuda:localrank --strict-device
```

- `--device` accepts `cpu`, `auto`, `cuda:N`, `cuda:localrank`, `sdaa:N`, and `sdaa:localrank`.
- `--local-rank` overrides rank discovery for `*:localrank`.
- `KMC_DEVICE` can provide the default device request when `--device` is omitted.
- `--strict-device` stops the run if the requested accelerator is unavailable.
- Without `--strict-device`, the workflow records fallback status in `outputs/tables/device_config.csv`.

设备通过同一个入口传入。`--local-rank` 可覆盖 `*:localrank` 的本地序号，`KMC_DEVICE` 可作为默认设备请求。`--strict-device` 会在请求的加速设备不可用时直接停止；不加该参数时，脚本会自动回退并把状态写入 `outputs/tables/device_config.csv`。

## Build Test Document

After running the workflow, rebuild the summary document:

```bash
python build_kmc_test_document.py
```

This writes:

- `outputs/reports/kmc_test_document.pdf`
- `outputs/reports/kmc_test_document.tex`
- `outputs/reports/kmc_test_document.md`

## Selected Generated Artifacts

- `outputs/cases/typical_cases.json`
- `outputs/tables/energy_results.csv`
- `outputs/tables/lattice_size_scan.csv`
- `outputs/tables/performance_records.csv`
- `outputs/tables/device_config.csv`
- `outputs/tables/module_timing_breakdown.csv`
- `outputs/tables/model_call_records.csv`
- `outputs/tables/stage_completion_matrix.csv`
- `outputs/tables/parallel_training_display.csv`
- `outputs/tables/multiscale_dataset.csv`
- `outputs/datasets/multiscale_dataset.csv`
- `outputs/datasets/kmc_snapshots.csv`
- `outputs/figures/material_evolution_curves.png`
- `outputs/figures/cu_cluster_structure.png`
- `outputs/figures/runtime_comparison.png`
- `outputs/tables/efficiency_comparison.csv`
- `outputs/reports/material_design_recommendations.md`
- `outputs/reports/output_audit_against_test_plan.md`
- `outputs/reports/acceptance_report.md`
- `outputs/reports/acceptance_report.tex`
- `outputs/reports/acceptance_report.pdf`
- `outputs/reports/kmc_test_document.tex`
- `outputs/reports/kmc_test_document.pdf`
- `outputs/manifest.json`

`outputs/manifest.json` records generated files except itself.

## Verification

```bash
python -m py_compile run_kmc_acceptance.py build_kmc_test_document.py
python run_kmc_acceptance.py --device cpu
python build_kmc_test_document.py
```

For report layout review, open the generated PDF files and inspect the first-page scale summary plus all figure pages.

## 中文说明

本文件夹包含 KMC 验收测试脚本、输出数据、图表和测试文档。核心流程是先运行 `run_kmc_acceptance.py` 生成 `outputs/`，再运行 `build_kmc_test_document.py` 生成汇报用 LaTeX/PDF 测试文档。默认扫描 10 个温度、7 个 Cu 含量、5 个 vacancy 含量，共 350 个温度/成分/缺陷组合；另做 10 个 lattice size 扫描，总计 360 个 KMC 算例，每组 100 个 KMC step，并行扩展性记录覆盖到 1024 节点。设备统一通过 `--device` 传入，支持 `cpu`、`cuda:localrank` 和 `sdaa:localrank` 等形式；设备解析结果会写入 `outputs/tables/device_config.csv`。
