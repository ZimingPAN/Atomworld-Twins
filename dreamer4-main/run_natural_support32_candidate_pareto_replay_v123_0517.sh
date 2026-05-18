#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
STAGE="${STAGE:-v123_candidate_pareto_replay_readonly}"
BASE_DIR="results/natural_teacher_support32_sequence_rollout_0517"
OUT_DIR="${BASE_DIR}/${STAGE}"

"${PYTHON_BIN}" diagnose_candidate_pareto_replay_v123.py \
  --v119-candidate-jsonl "${BASE_DIR}/v119_pair_joint_selector_grouped_readonly_smoke/candidate_joint_targets_v119.jsonl" \
  --v115-candidate-jsonl "${BASE_DIR}/v115_pair_interaction_distill_readonly/candidate_support_count_samples_v115.jsonl" \
  --v104-candidate-jsonl "${BASE_DIR}/v104_candidate_two_branch_selector_readonly_smoke20/candidate_two_branch_samples_v104.jsonl" \
  --v122-reference-summary "${BASE_DIR}/v122_candidate_pareto_selector_grouped_readonly/stage_summary.json" \
  --output-dir "${OUT_DIR}" \
  --ridge-l2 "${RIDGE_L2:-1.0}"
