#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v96_pair_level_vacancy_oracle_readonly}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
CACHE="${CACHE:-results/natural_teacher_support32_sequence_rollout_0517/v90_sequence_rollout_residual_smoke1/segments_v17.pt}"
OUT="${SAVE_DIR}/v96_pair_level_vacancy_oracle_readonly.json"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "cache=${CACHE}"
  echo "diagnostic=v96 read-only pair-level vacancy displacement upper bound"
} | tee -a "${LOG}"

"${PYTHON_BIN}" diagnose_vacancy_pair_oracle_v96.py \
  --cache "${CACHE}" \
  --output "${OUT}" \
  --stage-summary results/natural_teacher_support32_actionendpoint_0517/v85_action_edge_pair_smoke1/stage_summary.json \
  --stage-summary results/natural_teacher_support32_actionendpoint_0517/v88_edge_pair_multiobjective_smoke1/stage_summary.json \
  --stage-summary results/natural_teacher_support32_sequence_rollout_0517/v93_two_stage_vacancy_displacement_smoke20/stage_summary.json \
  --stage-summary results/natural_teacher_support32_sequence_rollout_0517/v95_terminal_typed_diff_support_smoke1b/stage_summary.json \
  --max-rows 24 2>&1 | tee -a "${LOG}"

cp "${OUT}" "${SAVE_DIR}/stage_summary.json"
echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
