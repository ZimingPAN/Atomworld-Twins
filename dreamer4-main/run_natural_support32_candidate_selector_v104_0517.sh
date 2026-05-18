#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v104_candidate_two_branch_selector_readonly_smoke20}"
INPUT="${INPUT:-${RESULT_ROOT}/v102_candidate_joint_ranking_readonly_smoke20/eval_long_vacpair_jointdiag20.json}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
LOG="${SAVE_DIR}/pipeline.log"
timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "input=${INPUT}"
  echo "mode=read-only two-branch candidate selector diagnostic and sample export"
} | tee -a "${LOG}"

"${PYTHON_BIN}" diagnose_candidate_joint_selector_v104.py \
  --input "${INPUT}" \
  --output "${SAVE_DIR}/candidate_two_branch_selector_v104.json" \
  --samples_output "${SAVE_DIR}/candidate_two_branch_samples_v104.jsonl" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
data = json.loads((save_dir / "candidate_two_branch_selector_v104.json").read_text(encoding="utf-8"))

def slim(summary):
    return {
        "count": summary.get("count"),
        "avg_site_f1": summary.get("avg_site_f1"),
        "avg_vacancy_pair_precision": summary.get("avg_vacancy_pair_precision"),
        "avg_vacancy_pair_recall": summary.get("avg_vacancy_pair_recall"),
        "avg_vacancy_pair_f1": summary.get("avg_vacancy_pair_f1"),
        "avg_teacher_reward_sum": summary.get("avg_teacher_reward_sum"),
        "avg_projected_changed_count": summary.get("avg_projected_changed_count"),
        "avg_vacancy_pair_selected_count": summary.get("avg_vacancy_pair_selected_count"),
    }

loo = data.get("leave_one_segment_out_two_branch", {}).get("pair_weight_sweep", {})
oracle = data.get("oracle_two_branch", {})
summary = {
    "stage": "${STAGE}",
    "input": "${INPUT}",
    "candidate_count": data.get("candidate_count"),
    "segment_count": data.get("segment_count"),
    "samples_output": data.get("samples_output"),
    "samples_count": data.get("samples_count"),
    "pair_selected_count_unique": data.get("pair_selected_count_unique"),
    "pair_selected_count_std": data.get("pair_selected_count_std"),
    "selected_by_planner": slim(data.get("selected_by_planner", {})),
    "oracle_energy_site": slim(data.get("oracle_energy_site", {})),
    "oracle_pair_precision": slim(data.get("oracle_pair_precision", {})),
    "oracle_two_branch_pair_weight_1": slim(oracle.get("1.0", {})),
    "loo_two_branch_pair_weight_0": slim(loo.get("0.0", {}).get("summary", {})),
    "loo_two_branch_pair_weight_0_5": slim(loo.get("0.5", {}).get("summary", {})),
    "loo_two_branch_pair_weight_1": slim(loo.get("1.0", {}).get("summary", {})),
    "selector_file": str(save_dir / "candidate_two_branch_selector_v104.json"),
}
(save_dir / "stage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
