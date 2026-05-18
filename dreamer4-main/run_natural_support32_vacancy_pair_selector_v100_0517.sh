#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v100_terminal_vacancy_pair_selector_readonly}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
CACHE="${CACHE:-results/natural_teacher_support32_sequence_rollout_0517/v95_sequence_rollout_orientation_fix_smoke1/segments_v17.pt}"
OUT="${SAVE_DIR}/v100_terminal_vacancy_pair_selector_readonly.json"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "cache=${CACHE}"
  echo "diagnostic=v100 read-only terminal vacancy-displacement pair selector target"
} | tee -a "${LOG}"

"${PYTHON_BIN}" diagnose_vacancy_pair_selector_v100.py \
  --cache "${CACHE}" \
  --output "${OUT}" \
  --max-rows 32 2>&1 | tee -a "${LOG}"

cp "${OUT}" "${SAVE_DIR}/stage_summary.json"
echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
