#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
SOURCE_STAGE="${SOURCE_STAGE:-v105_candidate_diverse_count_readonly_smoke20}"
STAGE="${STAGE:-v106_candidate_grouped_selector_readonly}"
SOURCE_DIR="${RESULT_ROOT}/${SOURCE_STAGE}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "source=${SOURCE_DIR}/candidate_diverse_samples_v105.jsonl"
  echo "mode=read-only grouped base-segment selector diagnostic"
} | tee -a "${LOG}"

"${PYTHON_BIN}" diagnose_candidate_joint_selector_v106.py \
  --samples "${SOURCE_DIR}/candidate_diverse_samples_v105.jsonl" \
  --output "${SAVE_DIR}/candidate_grouped_selector_v106.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
data = json.loads((save_dir / "candidate_grouped_selector_v106.json").read_text(encoding="utf-8"))

def slim(summary):
    return {
        "count": summary.get("count"),
        "avg_site_f1": summary.get("avg_site_f1"),
        "avg_vacancy_pair_precision": summary.get("avg_vacancy_pair_precision"),
        "avg_vacancy_pair_recall": summary.get("avg_vacancy_pair_recall"),
        "avg_vacancy_pair_f1": summary.get("avg_vacancy_pair_f1"),
        "avg_vacancy_pair_count_efficiency": summary.get("avg_vacancy_pair_count_efficiency"),
        "avg_teacher_reward_sum": summary.get("avg_teacher_reward_sum"),
        "avg_projected_changed_count": summary.get("avg_projected_changed_count"),
        "avg_vacancy_pair_selected_count": summary.get("avg_vacancy_pair_selected_count"),
    }

loo = data.get("grouped_leave_one_base_segment_out_two_branch", {}).get("pair_weight_sweep", {})
summary = {
    "stage": "${STAGE}",
    "source_stage": "${SOURCE_STAGE}",
    "sample_count": data.get("sample_count"),
    "base_segment_count": data.get("base_segment_count"),
    "pair_selected_count_unique": data.get("pair_selected_count_unique"),
    "pair_selected_count_std": data.get("pair_selected_count_std"),
    "planner_selected_expanded": slim(data.get("planner_selected_expanded", {})),
    "planner_selected_uncapped_only": slim(data.get("planner_selected_uncapped_only", {})),
    "oracle_energy_site_base": slim(data.get("oracle_energy_site_base", {})),
    "oracle_pair_precision_base": slim(data.get("oracle_pair_precision_base", {})),
    "oracle_two_branch_pair_weight_1": slim(data.get("oracle_two_branch_base", {}).get("1.0", {})),
    "grouped_loo_pair_weight_0": slim(loo.get("0.0", {}).get("summary", {})),
    "grouped_loo_pair_weight_0_sources": loo.get("0.0", {}).get("picked_source_histogram", {}),
    "grouped_loo_pair_weight_1": slim(loo.get("1.0", {}).get("summary", {})),
    "grouped_loo_pair_weight_1_sources": loo.get("1.0", {}).get("picked_source_histogram", {}),
    "grouped_loo_pair_weight_2": slim(loo.get("2.0", {}).get("summary", {})),
    "grouped_loo_pair_weight_2_sources": loo.get("2.0", {}).get("picked_source_histogram", {}),
    "selector_file": str(save_dir / "candidate_grouped_selector_v106.json"),
}
(save_dir / "stage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
