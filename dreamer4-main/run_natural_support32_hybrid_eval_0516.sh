#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-4}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_hybrid_0516}"
STAGE="${STAGE:-v67_edit_v65_rewardtau_v63_primaryproj}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
EDIT_CKPT="${EDIT_CKPT:-results/natural_teacher_support32_editonly_0516/v65_editonly_smoke1/final_model.pt}"
REWARD_TAU_CKPT="${REWARD_TAU_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
ROLLOUT_SEGMENTS="${ROLLOUT_SEGMENTS:-80}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "edit_checkpoint=${EDIT_CKPT}"
  echo "reward_tau_checkpoint=${REWARD_TAU_CKPT}"
  echo "diagnostic=primary edit/projection from v65; protected reward/gate/risk/tau heads from v63; auxiliary heads evaluate primary projected types"
} | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${EDIT_CKPT}" \
  --reward_checkpoint "${REWARD_TAU_CKPT}" \
  --duration_checkpoint "${REWARD_TAU_CKPT}" \
  --aux_projected_types_source primary \
  --planner_duration_checkpoint_source duration \
  --planner_segment_ks 8 16 32 \
  --rollout_segments "${ROLLOUT_SEGMENTS}" \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_hybrid80.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${EDIT_CKPT}" \
  --reward_checkpoint "${REWARD_TAU_CKPT}" \
  --duration_checkpoint "${REWARD_TAU_CKPT}" \
  --aux_projected_types_source primary \
  --planner_duration_checkpoint_source duration \
  --planner_segment_ks 8 16 32 \
  --rollout_segments "${ROLLOUT_SEGMENTS}" \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy \
  --planner_noop_risk_penalty 1.0 \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_energy_hybrid80.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = {
    "stage": "${STAGE}",
    "edit_checkpoint": "${EDIT_CKPT}",
    "reward_tau_checkpoint": "${REWARD_TAU_CKPT}",
    "aux_projected_types_source": "primary",
    "diagnostic": "hybrid protected branch eval: v65 edit/projection plus v63 reward/tau/risk on primary projected types",
    "files": {},
}
for name in ["eval_long_hybrid80.json", "eval_long_energy_hybrid80.json"]:
    path = save_dir / name
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    summary["files"][name] = {
        key: data[key]
        for key in [
            "reward_sum",
            "tau_expected",
            "cumulative",
            "chosen_k_histogram",
            "completed_rollout_segments",
            "requested_rollout_segments",
            "stop_reason",
            "reward_checkpoint",
            "duration_checkpoint",
            "aux_projected_types_source",
        ]
        if key in data
    }
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
