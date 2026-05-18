#!/usr/bin/env bash
set -euo pipefail

export STAGE="${STAGE:-v86_action_edge_pair_samesource_neg_smoke1}"
export INIT_CKPT="${INIT_CKPT:-results/natural_teacher_support32_actionendpoint_0517/v85_action_edge_pair_smoke1/final_model.pt}"
export PLANNER_CKPT="${PLANNER_CKPT:-results/natural_teacher_support32_actionendpoint_0517/v83_action_endpoint_smoke1b/final_model.pt}"
export CACHE="${CACHE:-results/natural_teacher_support32_actionendpoint_0517/v85_action_edge_pair_smoke1/segments.pt}"
export ACTION_EDGE_PAIR_NEGATIVE_MODE="${ACTION_EDGE_PAIR_NEGATIVE_MODE:-same_source_nn1}"
export ACTION_EDGE_PAIR_NEGATIVE_WEIGHT="${ACTION_EDGE_PAIR_NEGATIVE_WEIGHT:-2.0}"
export ACTION_EDGE_PAIR_RANK_MARGIN_WEIGHT="${ACTION_EDGE_PAIR_RANK_MARGIN_WEIGHT:-1.0}"

exec bash run_natural_support32_action_edge_pair_v85_0517.sh
