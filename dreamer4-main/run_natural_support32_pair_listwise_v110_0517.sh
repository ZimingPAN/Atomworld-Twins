#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v110_pair_listwise_contrastive_smoke1}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
INIT_CKPT="${INIT_CKPT:-results/natural_teacher_support32_sequence_rollout_0517/v109_structured_pair_hardneg_smoke1/final_model.pt}"
if [ ! -f "${INIT_CKPT}" ]; then
  INIT_CKPT="results/natural_teacher_support32_sequence_rollout_0517/v101b_terminal_vacancy_pair_selector_smoke1/final_model.pt"
fi
PLANNER_CKPT="${PLANNER_CKPT:-results/natural_teacher_support32_actionendpoint_0517/v83_action_endpoint_smoke1b/final_model.pt}"
PROTECTED_CKPT="${PROTECTED_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
CACHE="${CACHE:-results/natural_teacher_support32_sequence_rollout_0517/v101_terminal_vacancy_pair_selector_smoke1/segments_v18.pt}"
if [ ! -f "${CACHE}" ]; then
  CACHE="results/natural_teacher_support32_sequence_rollout_0517/v95_sequence_rollout_orientation_fix_smoke1/segments_v17.pt"
fi
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-3e-5}"
VACANCY_PAIR_NEGATIVE_COUNT="${VACANCY_PAIR_NEGATIVE_COUNT:-8}"
VACANCY_PAIR_STRUCTURED_NEGATIVE_COUNT="${VACANCY_PAIR_STRUCTURED_NEGATIVE_COUNT:-4}"
VACANCY_PAIR_LISTWISE_WEIGHT="${VACANCY_PAIR_LISTWISE_WEIGHT:-0.75}"
PRIOR_VACANCY_PAIR_LISTWISE_WEIGHT="${PRIOR_VACANCY_PAIR_LISTWISE_WEIGHT:-0.75}"
ROLLOUT_SEGMENTS="${ROLLOUT_SEGMENTS:-20}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "init=${INIT_CKPT}"
  echo "planner=${PLANNER_CKPT}"
  echo "protected_reward_tau=${PROTECTED_CKPT}"
  echo "cache=${CACHE}"
  echo "mode=terminal vacancy-pair true-vs-hard-negatives listwise contrastive smoke"
  echo "vacancy_pair_negative_count=${VACANCY_PAIR_NEGATIVE_COUNT}"
  echo "vacancy_pair_structured_negative_count=${VACANCY_PAIR_STRUCTURED_NEGATIVE_COUNT}"
  echo "vacancy_pair_listwise_weight=${VACANCY_PAIR_LISTWISE_WEIGHT}"
  echo "prior_vacancy_pair_listwise_weight=${PRIOR_VACANCY_PAIR_LISTWISE_WEIGHT}"
} | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_dreamer_macro_edit.py \
  --save_dir "${SAVE_DIR}" \
  --dataset_cache "${CACHE}" \
  --init_from "${INIT_CKPT}" \
  --planner_selected_from "${PLANNER_CKPT}" \
  --planner_selected_reward_checkpoint "${PROTECTED_CKPT}" \
  --planner_selected_duration_checkpoint "${PROTECTED_CKPT}" \
  --planner_selected_planner_duration_checkpoint_source duration \
  --planner_selected_aux_projected_types_source primary \
  --planner_selected_projection_change_source action_endpoint \
  --planner_selected_projection_topk_source action_endpoint \
  --planner_selected_projection_topk_budget 96 \
  --planner_selected_noop_risk_penalty 1.0 \
  --planner_selected_teacher_overlap_rerank_weight 0.5 \
  --planner_selected_store_candidate_overlap_masks \
  --planner_selected_min_projected_changed_sites 2 \
  --planner_selected_duration_source model \
  --planner_selected_tau_source model \
  --planner_selected_score_mode energy \
  --planner_selected_reward_prediction_source projected \
  --segment_ks 8 16 32 \
  --train_segments_per_k 4 \
  --val_segments_per_k 2 \
  --max_episode_steps 1024 \
  --max_segments_per_rollout 12 \
  --max_seed_vacancies 32 \
  --max_candidate_sites 2048 \
  --teacher_candidate_neighbor_depth 1 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --reward_prediction_source projected \
  --include_noop_segments \
  --keep_after_noop_segments \
  --train_vacancy_pair_heads_only \
  --vacancy_pair_weight 0.25 \
  --prior_vacancy_pair_weight 0.25 \
  --vacancy_pair_listwise_weight "${VACANCY_PAIR_LISTWISE_WEIGHT}" \
  --prior_vacancy_pair_listwise_weight "${PRIOR_VACANCY_PAIR_LISTWISE_WEIGHT}" \
  --vacancy_pair_semantic_weight 0.50 \
  --prior_vacancy_pair_semantic_weight 0.50 \
  --vacancy_pair_negative_count "${VACANCY_PAIR_NEGATIVE_COUNT}" \
  --vacancy_pair_structured_negative_count "${VACANCY_PAIR_STRUCTURED_NEGATIVE_COUNT}" \
  --action_edge_pair_weight 0.0 \
  --prior_action_edge_pair_weight 0.0 \
  --action_edge_pair_support_weight 0.0 \
  --prior_action_edge_pair_support_weight 0.0 \
  --action_edge_pair_semantic_weight 0.0 \
  --prior_action_edge_pair_semantic_weight 0.0 \
  --candidate_quality_weight 0.0 \
  --prior_candidate_quality_weight 0.0 \
  --proposal_support_weight 0.0 \
  --prior_proposal_support_weight 0.0 \
  --action_support_weight 0.0 \
  --prior_action_support_weight 0.0 \
  --action_source_support_weight 0.0 \
  --prior_action_source_support_weight 0.0 \
  --action_destination_support_weight 0.0 \
  --prior_action_destination_support_weight 0.0 \
  --reward_weight 0.0 \
  --prior_reward_weight 0.0 \
  --tau_weight 0.0 \
  --realized_tau_weight 0.0 \
  --tau_log_mu_weight 0.0 \
  --pair_weight 0.0 \
  --prior_pair_weight 0.0 \
  --prior_edit_weight 0.0 \
  --prior_latent_weight 0.0 \
  --proj_weight 0.0 \
  --path_weight 0.0 \
  --natural_teacher_backend kmc \
  --segment_boundary_mode adaptive_key_event \
  --adaptive_min_k 8 \
  --adaptive_candidate_horizon_source actual \
  --adaptive_key_moving_types 1 \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --device "${DEVICE}" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_time_alignment.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --cache "${CACHE}" \
  --split val \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_time_alignment_final.json" \
  --save_all_samples \
  2>&1 | tee -a "${LOG}"

run_long_eval() {
  local tag="$1"
  local projection_source="$2"
  local blend_alpha="$3"
  local output_path="${SAVE_DIR}/eval_long_${tag}_20.json"
  local rank_summary_path="${SAVE_DIR}/rank_summary_${tag}.json"
  {
    echo "=== eval ${tag} START $(timestamp) ==="
    echo "projection_source=${projection_source}"
    echo "blend_alpha=${blend_alpha}"
  } | tee -a "${LOG}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
    --checkpoint "${SAVE_DIR}/final_model.pt" \
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

run_long_eval "vacancy_rank_listwise" "vacancy_pair_completion" "0.5"
run_long_eval "energy_rank_listwise" "vacancy_pair_energy_blend_completion" "0.0"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
tags = ["vacancy_rank_listwise", "energy_rank_listwise"]

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def rank_pick(group):
    return {
        "avg_true_pair_rank_mean": group.get("avg_true_pair_rank_mean"),
        "avg_true_pair_rank_percentile_mean": group.get("avg_true_pair_rank_percentile_mean"),
        "avg_true_pair_mrr": group.get("avg_true_pair_mrr"),
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

summary = {
    "stage": "${STAGE}",
    "init_checkpoint": "${INIT_CKPT}",
    "planner_checkpoint": "${PLANNER_CKPT}",
    "protected_reward_tau_checkpoint": "${PROTECTED_CKPT}",
    "dataset_cache": "${CACHE}",
    "mode": "terminal vacancy-pair true-vs-hard-negatives listwise contrastive smoke",
    "vacancy_pair_negative_count": int("${VACANCY_PAIR_NEGATIVE_COUNT}"),
    "vacancy_pair_structured_negative_count": int("${VACANCY_PAIR_STRUCTURED_NEGATIVE_COUNT}"),
    "vacancy_pair_listwise_weight": float("${VACANCY_PAIR_LISTWISE_WEIGHT}"),
    "prior_vacancy_pair_listwise_weight": float("${PRIOR_VACANCY_PAIR_LISTWISE_WEIGHT}"),
    "rollout_segments_per_variant": int("${ROLLOUT_SEGMENTS}"),
    "files": {},
}
for path in ["metrics.json", "eval_time_alignment_final.json"]:
    p = save_dir / path
    if p.exists():
        data = load(p)
        summary["files"][path] = {
            key: data.get(key)
            for key in ["epoch", "train", "val", "final_val", "dataset", "selection_score"]
            if key in data
        }
for tag in tags:
    eval_data = load(save_dir / f"eval_long_{tag}_20.json")
    rank_data = load(save_dir / f"rank_summary_{tag}.json")
    selected = rank_data.get("selected_by_planner_rank", {})
    all_rank = rank_data.get("all_candidates_rank", {})
    summary["files"][f"eval_long_{tag}_20.json"] = {
        "completed_rollout_segments": eval_data.get("completed_rollout_segments"),
        "requested_rollout_segments": eval_data.get("requested_rollout_segments"),
        "stop_reason": eval_data.get("stop_reason"),
        "chosen_k_histogram": eval_data.get("chosen_k_histogram"),
        "cumulative": eval_data.get("cumulative", {}),
        "selected_site_f1": selected.get("avg_site_f1"),
        "selected_vacancy_pair_precision": selected.get("avg_vacancy_pair_precision"),
        "selected_vacancy_pair_recall": selected.get("avg_vacancy_pair_recall"),
        "selected_vacancy_pair_f1": selected.get("avg_vacancy_pair_f1"),
        "selected_pair_count": selected.get("avg_vacancy_pair_selected_count"),
        "selected_rank": rank_pick(selected),
        "all_candidate_rank": rank_pick(all_rank),
    }
(save_dir / "stage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== ${STAGE} DONE $(timestamp) ===" | tee -a "${LOG}"
