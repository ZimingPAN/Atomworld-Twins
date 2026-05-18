#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
ROLLOUT_SEGMENTS="${ROLLOUT_SEGMENTS:-20}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v107_pair_rank_readonly_smoke20}"
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

  echo "=== eval ${tag} END $(timestamp) ===" | tee -a "${LOG}"
}

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "checkpoint=${CHECKPOINT}"
  echo "protected_reward_tau=${PROTECTED_CKPT}"
  echo "rollout_segments=${ROLLOUT_SEGMENTS}"
  echo "mode=read-only vacancy-pair rank diagnostic"
} | tee -a "${LOG}"

run_eval "vacancy_rank_uncapped" "vacancy_pair_completion" "0.5"
run_eval "energy_rank_uncapped" "vacancy_pair_energy_blend_completion" "0.0"
run_eval "blend_rank_uncapped" "vacancy_pair_energy_blend_completion" "0.5"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")

def load_eval(tag):
    path = save_dir / f"eval_{tag}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return path, data

def slim_group(group):
    return {
        "avg_site_f1": group.get("avg_site_f1"),
        "avg_vacancy_pair_precision": group.get("avg_vacancy_pair_precision"),
        "avg_vacancy_pair_recall": group.get("avg_vacancy_pair_recall"),
        "avg_vacancy_pair_f1": group.get("avg_vacancy_pair_f1"),
        "avg_vacancy_pair_selected_count": group.get("avg_vacancy_pair_selected_count"),
        "avg_teacher_reward_sum": group.get("avg_teacher_reward_sum"),
        "avg_vacancy_pair_rank_found_recall": group.get("avg_vacancy_pair_rank_found_recall"),
        "avg_vacancy_pair_rank_mean": group.get("avg_vacancy_pair_rank_mean"),
        "avg_vacancy_pair_rank_median": group.get("avg_vacancy_pair_rank_median"),
        "avg_vacancy_pair_rank_percentile_mean": group.get("avg_vacancy_pair_rank_percentile_mean"),
        "avg_vacancy_pair_rank_typed_accuracy": group.get("avg_vacancy_pair_rank_typed_accuracy"),
        "avg_vacancy_pair_rank_recall_at_8": group.get("avg_vacancy_pair_rank_recall_at_8"),
        "avg_vacancy_pair_rank_recall_at_16": group.get("avg_vacancy_pair_rank_recall_at_16"),
        "avg_vacancy_pair_rank_recall_at_32": group.get("avg_vacancy_pair_rank_recall_at_32"),
        "avg_vacancy_pair_rank_recall_at_64": group.get("avg_vacancy_pair_rank_recall_at_64"),
        "avg_vacancy_pair_rank_recall_at_128": group.get("avg_vacancy_pair_rank_recall_at_128"),
        "avg_vacancy_pair_rank_recall_at_256": group.get("avg_vacancy_pair_rank_recall_at_256"),
        "avg_vacancy_pair_rank_recall_at_512": group.get("avg_vacancy_pair_rank_recall_at_512"),
        "avg_vacancy_pair_rank_recall_at_1024": group.get("avg_vacancy_pair_rank_recall_at_1024"),
    }

variants = {}
for tag in ["vacancy_rank_uncapped", "energy_rank_uncapped", "blend_rank_uncapped"]:
    path, data = load_eval(tag)
    candidate_joint = data.get("candidate_joint_diagnostic", {})
    variants[tag] = {
        "file": str(path),
        "completed_rollout_segments": data.get("completed_rollout_segments"),
        "requested_rollout_segments": data.get("requested_rollout_segments"),
        "stop_reason": data.get("stop_reason"),
        "chosen_k_histogram": data.get("chosen_k_histogram"),
        "cumulative": data.get("cumulative", {}),
        "selected_by_planner": slim_group(candidate_joint.get("selected_by_planner", {})),
        "all_candidates": slim_group(candidate_joint.get("all_candidates", {})),
        "selector_site_f1": slim_group(
            candidate_joint.get("selector_upper_bounds", {}).get("site_f1", {})
        ),
        "selector_vacancy_pair_f1": slim_group(
            candidate_joint.get("selector_upper_bounds", {}).get("vacancy_pair_f1", {})
        ),
        "selector_teacher_reward": slim_group(
            candidate_joint.get("selector_upper_bounds", {}).get("teacher_reward_sum", {})
        ),
        "segment_preview": candidate_joint.get("segment_preview", [])[:3],
    }

summary = {
    "stage": "${STAGE}",
    "checkpoint": "${CHECKPOINT}",
    "protected_reward_tau_checkpoint": "${PROTECTED_CKPT}",
    "rollout_segments_per_variant": int("${ROLLOUT_SEGMENTS}"),
    "mode": "read-only vacancy-pair ranked-list target diagnostic",
    "variants": variants,
}
(save_dir / "stage_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
