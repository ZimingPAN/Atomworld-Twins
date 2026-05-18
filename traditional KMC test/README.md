# KMC Acceptance Test

Run the KMC acceptance workflow with:

```bash
python run_kmc_acceptance.py
```

All generated files are written under `outputs/`.

Device selection uses one interface:

```bash
python run_kmc_acceptance.py --device cpu
python run_kmc_acceptance.py --device cuda:localrank
python run_kmc_acceptance.py --device sdaa:localrank
python run_kmc_acceptance.py --device cuda:localrank --strict-device
```

`--strict-device` stops the run if the requested accelerator is unavailable; without it the workflow records the fallback status in `outputs/tables/device_config.csv`.

设备通过同一个入口传入。`--strict-device` 会在请求的加速设备不可用时直接停止；不加该参数时，脚本会自动回退并把状态写入 `outputs/tables/device_config.csv`。

Main generated artifacts:

- `outputs/cases/typical_cases.json`
- `outputs/tables/energy_results.csv`
- `outputs/tables/performance_records.csv`
- `outputs/tables/device_config.csv`
- `outputs/tables/module_timing_breakdown.csv`
- `outputs/tables/model_call_records.csv`
- `outputs/tables/stage_completion_matrix.csv`
- `outputs/datasets/multiscale_dataset.csv`
- `outputs/figures/material_evolution_curves.png`
- `outputs/figures/cu_cluster_structure.png`
- `outputs/tables/efficiency_comparison.csv`
- `outputs/reports/material_design_recommendations.md`
- `outputs/reports/output_audit_against_test_plan.md`
- `outputs/reports/acceptance_report.md`
- `outputs/reports/acceptance_report.docx`
