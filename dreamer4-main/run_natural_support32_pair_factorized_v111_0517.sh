#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v111_pair_factorized_readonly_smoke20}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
CHECKPOINT="${CHECKPOINT:-results/natural_teacher_support32_sequence_rollout_0517/v110_pair_listwise_contrastive_smoke1/final_model.pt}"
if [ ! -f "${CHECKPOINT}" ]; then
  CHECKPOINT="results/natural_teacher_support32_sequence_rollout_0517/v101b_terminal_vacancy_pair_selector_smoke1/final_model.pt"
fi
PROTECTED_CKPT="${PROTECTED_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
ROLLOUT_SEGMENTS="${ROLLOUT_SEGMENTS:-20}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

{
  echo "=== ${STAGE} START $(timestamp) ==="
  echo "checkpoint=${CHECKPOINT}"
  echo "protected_reward_tau=${PROTECTED_CKPT}"
  echo "mode=eval-only factorized vacancy-pair decomposition diagnostic"
} | tee -a "${LOG}"

run_factorized_eval() {
  local tag="$1"
  local projection_source="$2"
  local blend_alpha="$3"
  local output_path="${SAVE_DIR}/eval_long_${tag}_20.json"
  local rank_summary_path="${SAVE_DIR}/rank_summary_${tag}.json"
  local factorized_summary_path="${SAVE_DIR}/factorized_summary_${tag}.json"

  echo "=== eval ${tag} START $(timestamp) ===" | tee -a "${LOG}"
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
    --planner_vacancy_pair_factorized_diagnostic \
    --planner_vacancy_pair_factorized_max_pairs 0 \
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

  "${PYTHON_BIN}" diagnose_vacancy_pair_factorized_v111.py \
    --eval-json "${output_path}" \
    --output "${factorized_summary_path}" \
    2>&1 | tee -a "${LOG}"

  echo "=== eval ${tag} END $(timestamp) ===" | tee -a "${LOG}"
}

run_factorized_eval "vacancy_factorized" "vacancy_pair_completion" "0.5"
run_factorized_eval "energy_factorized" "vacancy_pair_energy_blend_completion" "0.0"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
tags = ["vacancy_factorized", "energy_factorized"]

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def slim_factor(group):
    if not isinstance(group, dict):
        return {}
    factors = group.get("factor_rank_summaries", {})
    compact = {}
    for name, payload in factors.items():
        if not isinstance(payload, dict):
            continue
        recall = payload.get("recall_at_rank", {})
        compact[name] = {
            "rank_percentile": payload.get("true_pair_rank_percentile_mean"),
            "recall128": recall.get("128") if isinstance(recall, dict) else None,
            "false_positive128": (
                payload.get("topk_false_positive_rate", {}).get("128")
                if isinstance(payload.get("topk_false_positive_rate"), dict)
                else None
            ),
        }
    return {
        "candidate_count": group.get("candidate_count"),
        "avg_site_f1": group.get("avg_site_f1"),
        "avg_vacancy_pair_f1": group.get("avg_vacancy_pair_f1"),
        "best_factor_by_recall_at_128": group.get("best_factor_by_recall_at_128"),
        "best_factor_by_rank_percentile": group.get("best_factor_by_rank_percentile"),
        "factors": compact,
    }

summary = {
    "stage": "${STAGE}",
    "checkpoint": "${CHECKPOINT}",
    "protected_reward_tau": "${PROTECTED_CKPT}",
    "mode": "eval-only v111 factorized pair decomposition",
    "results": {},
}
for tag in tags:
    eval_data = load(save_dir / f"eval_long_{tag}_20.json")
    rank_data = load(save_dir / f"rank_summary_{tag}.json")
    factor_data = load(save_dir / f"factorized_summary_{tag}.json")
    selected_rank = rank_data.get("selected_by_planner_rank", {})
    summary["results"][tag] = {
        "completed_segments": eval_data.get("completed_rollout_segments"),
        "stop_reason": eval_data.get("stop_reason"),
        "chosen_k_histogram": eval_data.get("chosen_k_histogram"),
        "cumulative": eval_data.get("cumulative"),
        "tau_expected": eval_data.get("tau_expected"),
        "selected_site_f1": selected_rank.get("avg_site_f1"),
        "selected_pair_precision": selected_rank.get("avg_vacancy_pair_precision"),
        "selected_pair_recall": selected_rank.get("avg_vacancy_pair_recall"),
        "selected_pair_f1": selected_rank.get("avg_vacancy_pair_f1"),
        "selected_rank_recall128": (
            selected_rank.get("avg_recall_at_rank", {}).get("128")
            if isinstance(selected_rank.get("avg_recall_at_rank"), dict)
            else None
        ),
        "selected_factorized": slim_factor(factor_data.get("selected_by_planner")),
        "all_factorized": slim_factor(factor_data.get("all_candidates")),
    }

out = save_dir / "stage_summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== ${STAGE} END $(timestamp) ===" | tee -a "${LOG}"
