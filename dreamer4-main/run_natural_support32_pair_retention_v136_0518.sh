#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
STAGE="${STAGE:-v136_pair_retention_readonly}"
ROOT="results/natural_teacher_support32_sequence_rollout_0517"
OUT_DIR="${ROOT}/${STAGE}"

"${PYTHON_BIN}" diagnose_v135_pair_retention_v136.py \
  --diagnostic-eval "${ROOT}/v135_guarded_budget_projection_diagnostic_smoke20/eval_long_guarded_budget_projection_diagnostic_20.json" \
  --replace-eval "${ROOT}/v135_guarded_budget_projection_replace_retry_smoke20/eval_long_guarded_budget_projection_replace_20.json" \
  --output-dir "${OUT_DIR}"
