#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/home/likun-share/panziming/pydeps:/home/likun-share/panziming/AtomWorld-Mirror/kmcteacher_backend:/home/likun-share/panziming/AtomWorld-Mirror/LightZero-main:/home/likun-share/panziming/AtomWorld-Mirror/dreamer4-main:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_sequence_rollout_0517}"
STAGE="${STAGE:-v102_candidate_joint_ranking_readonly_smoke20}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
CHECKPOINT="${CHECKPOINT:-results/natural_teacher_support32_sequence_rollout_0517/v101b_terminal_vacancy_pair_selector_smoke1/final_model.pt}"
PROTECTED_CKPT="${PROTECTED_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "checkpoint=${CHECKPOINT}"
  echo "protected_reward_tau=${PROTECTED_CKPT}"
  echo "mode=read-only candidate-level joint-ranking diagnostic"
} | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${CHECKPOINT}" \
  --reward_checkpoint "${PROTECTED_CKPT}" \
  --duration_checkpoint "${PROTECTED_CKPT}" \
  --aux_projected_types_source primary \
  --planner_segment_ks 8 16 32 \
  --rollout_segments 20 \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy \
  --planner_noop_risk_penalty 1.0 \
  --planner_projection_change_source vacancy_pair_completion \
  --planner_projection_topk_source none \
  --planner_projection_topk_budget 0 \
  --planner_edge_completion_anchor_source action_source \
  --planner_edge_completion_destination_source action_destination \
  --planner_edge_completion_anchor_budget 32 \
  --planner_edge_completion_destinations_per_anchor 4 \
  --planner_edge_completion_destination_scope global_atom \
  --planner_edge_completion_global_pair_budget 0 \
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
  --output "${SAVE_DIR}/eval_long_vacpair_jointdiag20.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
eval_path = save_dir / "eval_long_vacpair_jointdiag20.json"
data = json.loads(eval_path.read_text())

def avg(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / max(len(values), 1)

site_f1 = []
pair_f1 = []
pair_precision = []
pair_recall = []
for seg in data.get("segments", []):
    po = seg.get("proposal_overlap")
    if isinstance(po, dict):
        site_f1.append(float(po.get("f1", 0.0)))
    vpo = seg.get("vacancy_pair_overlap")
    if isinstance(vpo, dict):
        pair_f1.append(float(vpo.get("f1", 0.0)))
        pair_precision.append(float(vpo.get("precision", 0.0)))
        pair_recall.append(float(vpo.get("recall", 0.0)))

summary = {
    "stage": "${STAGE}",
    "checkpoint": "${CHECKPOINT}",
    "protected_reward_tau_checkpoint": "${PROTECTED_CKPT}",
    "eval_file": str(eval_path),
    "completed_rollout_segments": data.get("completed_rollout_segments"),
    "requested_rollout_segments": data.get("requested_rollout_segments"),
    "stop_reason": data.get("stop_reason"),
    "chosen_k_histogram": data.get("chosen_k_histogram"),
    "cumulative": data.get("cumulative"),
    "tau_expected": data.get("tau_expected"),
    "teacher_overlap_oracle": data.get("teacher_overlap_oracle"),
    "selected_site_f1_avg": avg(site_f1),
    "selected_vacancy_pair_precision_avg": avg(pair_precision),
    "selected_vacancy_pair_recall_avg": avg(pair_recall),
    "selected_vacancy_pair_f1_avg": avg(pair_f1),
    "candidate_joint_diagnostic": data.get("candidate_joint_diagnostic", {}),
}
(save_dir / "stage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "=== ${STAGE} END $(date -Is) ===" | tee -a "${LOG}"
