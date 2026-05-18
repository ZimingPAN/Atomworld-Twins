#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
STAGE="${STAGE:-v130_recall_floor_selector_replay_readonly}"
ROOT_DIR="results/natural_teacher_support32_sequence_rollout_0517"
SAVE_DIR="${ROOT_DIR}/${STAGE}"

mkdir -p "${SAVE_DIR}"

{
  echo "[v130] stage=${STAGE}"
  echo "[v130] python=${PYTHON_BIN}"
  "${PYTHON_BIN}" diagnose_candidate_recall_floor_replay_v130.py \
    --output-dir "${SAVE_DIR}"
} 2>&1 | tee "${SAVE_DIR}/pipeline.log"
