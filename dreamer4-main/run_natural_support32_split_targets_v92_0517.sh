#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/likun-share/panziming/AtomWorld-Mirror/.venv/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

RESULT_ROOT="results/natural_teacher_support32_sequence_rollout_0517"
STAGE="${STAGE:-v92_split_target_readonly}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
LOG="${SAVE_DIR}/pipeline.log"
CACHE="${CACHE:-${RESULT_ROOT}/v90_sequence_rollout_residual_smoke1/segments_v17.pt}"

mkdir -p "${SAVE_DIR}"
{
  echo "[v92] START $(date -Is)"
  echo "[v92] CACHE=${CACHE}"
  echo "[v92] SAVE_DIR=${SAVE_DIR}"
  "${PYTHON_BIN}" diagnose_split_targets_v92.py \
    --cache "${CACHE}" \
    --output_dir "${SAVE_DIR}"
  echo "[v92] DONE $(date -Is)"
} 2>&1 | tee -a "${LOG}"
