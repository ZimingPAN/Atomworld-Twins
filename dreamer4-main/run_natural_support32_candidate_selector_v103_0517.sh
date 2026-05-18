#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v103_candidate_joint_selector_readonly_smoke20}"
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
  echo "mode=read-only candidate-level listwise/calibrated-count selector diagnostic"
} | tee -a "${LOG}"

"${PYTHON_BIN}" diagnose_candidate_joint_selector_v103.py \
  --input "${INPUT}" \
  --output "${SAVE_DIR}/candidate_joint_selector_v103.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
data = json.loads((save_dir / "candidate_joint_selector_v103.json").read_text(encoding="utf-8"))

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
        "avg_v103_joint_target": summary.get("avg_v103_joint_target"),
    }

selectors = data.get("selectors", {})
loo = data.get("leave_one_segment_out", {})
summary = {
    "stage": "${STAGE}",
    "input": "${INPUT}",
    "candidate_count": data.get("candidate_count"),
    "segment_count": data.get("segment_count"),
    "selected_by_planner": slim(selectors.get("selected_by_planner", {})),
    "oracle_v103_joint_target": slim(selectors.get("oracle_v103_joint_target", {})),
    "oracle_site_f1": slim(selectors.get("oracle_site_f1", {})),
    "oracle_teacher_reward_sum": slim(selectors.get("oracle_teacher_reward_sum", {})),
    "oracle_vacancy_pair_f1": slim(selectors.get("oracle_vacancy_pair_f1", {})),
    "loo_joint_target": slim(loo.get("joint_target", {}).get("summary", {})),
    "loo_precision_target": slim(loo.get("precision_target", {}).get("summary", {})),
    "selector_file": str(save_dir / "candidate_joint_selector_v103.json"),
}
(save_dir / "stage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
