#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
ROLLOUT_SEGMENTS="${ROLLOUT_SEGMENTS:-20}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v105_candidate_diverse_count_readonly_smoke20}"
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
  local anchor_budget="$3"
  local destinations_per_anchor="$4"
  local global_pair_budget="$5"
  local output_path="${SAVE_DIR}/eval_${tag}.json"

  {
    echo "=== eval ${tag} START $(timestamp) ==="
    echo "projection_source=${projection_source}"
    echo "anchor_budget=${anchor_budget}"
    echo "destinations_per_anchor=${destinations_per_anchor}"
    echo "global_pair_budget=${global_pair_budget}"
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
    --planner_edge_completion_anchor_budget "${anchor_budget}" \
    --planner_edge_completion_destinations_per_anchor "${destinations_per_anchor}" \
    --planner_edge_completion_destination_scope global_atom \
    --planner_edge_completion_global_pair_budget "${global_pair_budget}" \
    --planner_projection_change_blend_alpha 0.5 \
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

  echo "=== eval ${tag} END $(timestamp) ===" | tee -a "${LOG}"
}

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "checkpoint=${CHECKPOINT}"
  echo "protected_reward_tau=${PROTECTED_CKPT}"
  echo "rollout_segments=${ROLLOUT_SEGMENTS}"
  echo "mode=read-only diverse candidate/count collection"
} | tee -a "${LOG}"

run_eval "uncapped_a32_d4" "vacancy_pair_completion" 32 4 0
run_eval "kbudget_a32_d4" "vacancy_pair_completion" 32 4 -1
run_eval "budget16_a32_d4" "vacancy_pair_completion" 32 4 16
run_eval "budget32_a32_d4" "vacancy_pair_completion" 32 4 32
run_eval "budget64_a32_d4" "vacancy_pair_completion" 32 4 64
run_eval "energyblend_kbudget_a32_d4" "vacancy_pair_energy_blend_completion" 32 4 -1

"${PYTHON_BIN}" diagnose_candidate_joint_selector_v105.py \
  --input "uncapped_a32_d4=${SAVE_DIR}/eval_uncapped_a32_d4.json" \
  --input "kbudget_a32_d4=${SAVE_DIR}/eval_kbudget_a32_d4.json" \
  --input "budget16_a32_d4=${SAVE_DIR}/eval_budget16_a32_d4.json" \
  --input "budget32_a32_d4=${SAVE_DIR}/eval_budget32_a32_d4.json" \
  --input "budget64_a32_d4=${SAVE_DIR}/eval_budget64_a32_d4.json" \
  --input "energyblend_kbudget_a32_d4=${SAVE_DIR}/eval_energyblend_kbudget_a32_d4.json" \
  --output "${SAVE_DIR}/candidate_diverse_selector_v105.json" \
  --samples_output "${SAVE_DIR}/candidate_diverse_samples_v105.jsonl" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
data = json.loads((save_dir / "candidate_diverse_selector_v105.json").read_text(encoding="utf-8"))

def slim(summary):
    return {
        "count": summary.get("count"),
        "avg_site_f1": summary.get("avg_site_f1"),
        "avg_vacancy_pair_precision": summary.get("avg_vacancy_pair_precision"),
        "avg_vacancy_pair_recall": summary.get("avg_vacancy_pair_recall"),
        "avg_vacancy_pair_f1": summary.get("avg_vacancy_pair_f1"),
        "avg_vacancy_pair_count_efficiency": summary.get("avg_vacancy_pair_count_efficiency"),
        "avg_teacher_reward_sum": summary.get("avg_teacher_reward_sum"),
        "avg_projected_changed_count": summary.get("avg_projected_changed_count"),
        "avg_vacancy_pair_selected_count": summary.get("avg_vacancy_pair_selected_count"),
    }

loo = data.get("leave_one_segment_out_two_branch", {}).get("pair_weight_sweep", {})
oracle = data.get("oracle_two_branch", {})
summary = {
    "stage": "${STAGE}",
    "checkpoint": "${CHECKPOINT}",
    "protected_reward_tau_checkpoint": "${PROTECTED_CKPT}",
    "rollout_segments_per_variant": int("${ROLLOUT_SEGMENTS}"),
    "source_count": len(data.get("inputs", [])),
    "candidate_count": data.get("candidate_count"),
    "segment_count": data.get("segment_count"),
    "samples_output": data.get("samples_output"),
    "samples_count": data.get("samples_count"),
    "pair_selected_count_unique": data.get("pair_selected_count_unique"),
    "pair_selected_count_std": data.get("pair_selected_count_std"),
    "selected_by_planner": slim(data.get("selected_by_planner", {})),
    "oracle_energy_site": slim(data.get("oracle_energy_site", {})),
    "oracle_pair_precision": slim(data.get("oracle_pair_precision", {})),
    "oracle_two_branch_pair_weight_1": slim(oracle.get("1.0", {})),
    "loo_two_branch_pair_weight_0": slim(loo.get("0.0", {}).get("summary", {})),
    "loo_two_branch_pair_weight_0_5": slim(loo.get("0.5", {}).get("summary", {})),
    "loo_two_branch_pair_weight_1": slim(loo.get("1.0", {}).get("summary", {})),
    "loo_two_branch_pair_weight_2": slim(loo.get("2.0", {}).get("summary", {})),
    "selector_file": str(save_dir / "candidate_diverse_selector_v105.json"),
}
(save_dir / "stage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
