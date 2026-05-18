#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
STAGE="${STAGE:-v124_candidate_feature_parity_readonly}"
BASE_DIR="${BASE_DIR:-results/natural_teacher_support32_sequence_rollout_0517}"

"${PYTHON_BIN}" diagnose_candidate_feature_parity_v124.py \
  --selector-spec "${BASE_DIR}/v123_candidate_pareto_replay_readonly/candidate_pareto_selector_spec_v123.json" \
  --v119-candidate-jsonl "${BASE_DIR}/v119_pair_joint_selector_grouped_readonly_smoke/candidate_joint_targets_v119.jsonl" \
  --v104-candidate-jsonl "${BASE_DIR}/v104_candidate_two_branch_selector_readonly_smoke20/candidate_two_branch_samples_v104.jsonl" \
  --eval-source "eval_macro_long_trajectory.py" \
  --output-dir "${BASE_DIR}/${STAGE}"
