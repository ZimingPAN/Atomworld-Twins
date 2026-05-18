#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
ROLLOUT_SEGMENTS="${ROLLOUT_SEGMENTS:-20}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v108_pair_rank_hardneg_readonly_smoke20}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
CHECKPOINT="${CHECKPOINT:-results/natural_teacher_support32_sequence_rollout_0517/v101b_terminal_vacancy_pair_selector_smoke1/final_model.pt}"
PROTECTED_CKPT="${PROTECTED_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

run_eval() {
  local tag="$1"
  local projection_source="$2"
  local blend_alpha="$3"
  local output_path="${SAVE_DIR}/eval_${tag}.json"
  local rank_summary_path="${SAVE_DIR}/rank_summary_${tag}.json"

  {
    echo "=== eval ${tag} START $(timestamp) ==="
    echo "projection_source=${projection_source}"
    echo "blend_alpha=${blend_alpha}"
  } | tee -a "${LOG}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
    --checkpoint "${CHECKPOINT}" \
    --reward_checkpoint "${PROTECTED_CKPT}" \
    --duration_checkpoint "${PROTECTED_CKPT}" \
    --aux_projected_types_source primary \
    --planner_segment_ks 8 16 32 \
    --rollout_segments "${ROLLOUT_SEGMENTS}" \
    --max_episode_steps_override 1024 \
    --duration_source model \
    --planner_tau_source model \
    --planner_score_mode energy \
    --planner_noop_risk_penalty 1.0 \
    --planner_projection_change_source "${projection_source}" \
    --planner_projection_topk_source none \
    --planner_projection_topk_budget 0 \
    --planner_edge_completion_anchor_source action_source \
    --planner_edge_completion_destination_source action_destination \
    --planner_edge_completion_anchor_budget 32 \
    --planner_edge_completion_destinations_per_anchor 4 \
    --planner_edge_completion_destination_scope global_atom \
    --planner_edge_completion_global_pair_budget 0 \
    --planner_projection_change_blend_alpha "${blend_alpha}" \
    --planner_edge_pair_multiobjective_type_weight 0.15 \
    --planner_edge_pair_multiobjective_order_weight 0.10 \
    --planner_vacancy_pair_rank_diagnostic \
    --planner_vacancy_pair_rank_max_pairs 0 \
    --proposal_diagnostic \
    --proposal_diagnostic_max_sites 384 \
    --planner_teacher_overlap_oracle_mode add \
    --planner_teacher_overlap_oracle_weight 0.0 \
    --planner_teacher_overlap_oracle_metric overlap_reward_norm \
    --planner_candidate_joint_diagnostic \
    --planner_candidate_joint_compact_candidates \
    --min_projected_changed_sites 2 \
    --print_segments 0 \
    --progress_every 5 \
    --device "${DEVICE}" \
    --output "${output_path}" \
    2>&1 | tee -a "${LOG}"

  "${PYTHON_BIN}" diagnose_vacancy_pair_rank_v108.py \
    --eval-json "${output_path}" \
    --output "${rank_summary_path}" \
    2>&1 | tee -a "${LOG}"

  echo "=== eval ${tag} END $(timestamp) ===" | tee -a "${LOG}"
}

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "checkpoint=${CHECKPOINT}"
  echo "protected_reward_tau=${PROTECTED_CKPT}"
  echo "rollout_segments=${ROLLOUT_SEGMENTS}"
  echo "mode=read-only vacancy-pair hard-negative composition diagnostic"
} | tee -a "${LOG}"

run_eval "vacancy_rank_uncapped" "vacancy_pair_completion" "0.5"
run_eval "energy_rank_uncapped" "vacancy_pair_energy_blend_completion" "0.0"
run_eval "blend_rank_uncapped" "vacancy_pair_energy_blend_completion" "0.5"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
tags = ["vacancy_rank_uncapped", "energy_rank_uncapped", "blend_rank_uncapped"]

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def pick_topk(group):
    return {
        "avg_true_pair_found_recall": group.get("avg_true_pair_found_recall"),
        "avg_true_pair_rank_mean": group.get("avg_true_pair_rank_mean"),
        "avg_true_pair_rank_percentile_mean": group.get("avg_true_pair_rank_percentile_mean"),
        "avg_true_pair_mrr": group.get("avg_true_pair_mrr"),
        "avg_true_pair_typed_rank_accuracy": group.get("avg_true_pair_typed_rank_accuracy"),
        "avg_recall_at_rank": group.get("avg_recall_at_rank"),
        "avg_topk_false_positive_rate": group.get("avg_topk_false_positive_rate"),
        "avg_topk_true_pair_count": group.get("avg_topk_true_pair_count"),
        "avg_topk_source_hard_negative_count": group.get("avg_topk_source_hard_negative_count"),
        "avg_topk_destination_hard_negative_count": group.get("avg_topk_destination_hard_negative_count"),
        "avg_topk_source_destination_unpaired_count": group.get("avg_topk_source_destination_unpaired_count"),
        "avg_topk_type_mismatch_count": group.get("avg_topk_type_mismatch_count"),
        "avg_topk_true_score_mean": group.get("avg_topk_true_score_mean"),
        "avg_topk_false_score_mean": group.get("avg_topk_false_score_mean"),
    }

variants = {}
for tag in tags:
    eval_data = load_json(save_dir / f"eval_{tag}.json")
    rank_data = load_json(save_dir / f"rank_summary_{tag}.json")
    selected_rank = rank_data.get("selected_by_planner_rank", {})
    all_rank = rank_data.get("all_candidates_rank", {})
    variants[tag] = {
        "completed_rollout_segments": eval_data.get("completed_rollout_segments"),
        "requested_rollout_segments": eval_data.get("requested_rollout_segments"),
        "stop_reason": eval_data.get("stop_reason"),
        "chosen_k_histogram": eval_data.get("chosen_k_histogram"),
        "cumulative": eval_data.get("cumulative", {}),
        "selected_site_f1": selected_rank.get("avg_site_f1"),
        "selected_vacancy_pair_precision": selected_rank.get("avg_vacancy_pair_precision"),
        "selected_vacancy_pair_recall": selected_rank.get("avg_vacancy_pair_recall"),
        "selected_vacancy_pair_f1": selected_rank.get("avg_vacancy_pair_f1"),
        "selected_pair_count": selected_rank.get("avg_vacancy_pair_selected_count"),
        "selected_teacher_reward_sum": selected_rank.get("avg_teacher_reward_sum"),
        "selected_rank_hardneg": pick_topk(selected_rank),
        "all_candidate_rank_hardneg": pick_topk(all_rank),
        "selector_upper_bounds": rank_data.get("candidate_joint_summary", {}).get("selector_upper_bounds", {}),
    }

summary = {
    "stage": "${STAGE}",
    "checkpoint": "${CHECKPOINT}",
    "protected_reward_tau_checkpoint": "${PROTECTED_CKPT}",
    "rollout_segments_per_variant": int("${ROLLOUT_SEGMENTS}"),
    "mode": "read-only vacancy-pair hard-negative composition diagnostic",
    "variants": variants,
}
(save_dir / "stage_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
