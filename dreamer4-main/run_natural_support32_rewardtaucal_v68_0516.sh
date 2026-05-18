#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-5}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_rewardtaucal_0516}"
STAGE="${STAGE:-v68_v64edit_v63heads_rewardtaucal}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
V64_CKPT="${V64_CKPT:-results/natural_teacher_support32_dualsupport_0516/v64_teacheraug_positive_riskguard/final_model.pt}"
V63_CKPT="${V63_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
PLANNER_CKPT="${PLANNER_CKPT:-results/natural_teacher_support32_nooprisk_0516/v63_nooprisk_head_plannerpenalty/final_model.pt}"
CACHE="${CACHE:-results/natural_teacher_support32_dualsupport_0516/v64_teacheraug_positive_riskguard/segments.pt}"
MERGED_CKPT="${MERGED_CKPT:-${SAVE_DIR}/v64_with_v63_rewardtau_heads.pt}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-5e-5}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "v64_edit_projection_checkpoint=${V64_CKPT}"
  echo "v63_protected_reward_tau_checkpoint=${V63_CKPT}"
  echo "cache=${CACHE}"
  echo "diagnostic=merge v63 reward/gate/duration/risk heads into v64 edit/projection branch, then train reward+duration heads only"
} | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY 2>&1 | tee -a "${LOG}"
import copy
from pathlib import Path
import torch

v64_path = Path("${V64_CKPT}")
v63_path = Path("${V63_CKPT}")
out_path = Path("${MERGED_CKPT}")
v64 = torch.load(v64_path, map_location="cpu", weights_only=False)
v63 = torch.load(v63_path, map_location="cpu", weights_only=False)
merged = copy.deepcopy(v64)
prefixes = (
    "reward_head.",
    "reward_context_head.",
    "reward_gate_head.",
    "reward_gate_context_head.",
    "duration_head.",
    "realized_duration_head.",
    "duration_context_head.",
    "realized_duration_context_head.",
    "noop_risk_head.",
    "noop_risk_context_head.",
)
copied = []
skipped = []
for key, value in v63["model"].items():
    if not key.startswith(prefixes):
        continue
    if key in merged["model"] and tuple(merged["model"][key].shape) == tuple(value.shape):
        merged["model"][key] = value.detach().clone()
        copied.append(key)
    else:
        skipped.append(key)
merged.setdefault("args", {})
merged["args"] = dict(merged.get("args", {}))
merged["args"]["merged_reward_tau_heads_from"] = str(v63_path)
merged["args"]["merged_edit_projection_from"] = str(v64_path)
merged["optimizer"] = {}
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(merged, out_path)
print({"merged_checkpoint": str(out_path), "copied_head_tensors": len(copied), "skipped": skipped})
PY

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_dreamer_macro_edit.py \
  --save_dir "${SAVE_DIR}" \
  --dataset_cache "${CACHE}" \
  --init_from "${MERGED_CKPT}" \
  --planner_selected_from "${PLANNER_CKPT}" \
  --planner_selected_min_projected_changed_sites 2 \
  --planner_selected_duration_source baseline \
  --planner_selected_tau_source baseline \
  --planner_selected_score_mode energy_per_sqrt_tau \
  --planner_selected_reward_prediction_source projected \
  --segment_ks 8 16 32 \
  --train_segments_per_k 192 \
  --val_segments_per_k 48 \
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
  --train_reward_duration_heads_only \
  --realized_tau_weight 0.05 \
  --tau_weight 1.0 \
  --tau_log_mu_weight 0.75 \
  --reward_weight 1.2 \
  --prior_reward_weight 1.2 \
  --reward_gate_weight 4.0 \
  --reward_zero_weight 8.0 \
  --reward_sign_weight 0.5 \
  --reward_magnitude_weight 1.0 \
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

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 8 16 32 \
  --rollout_segments 80 \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_trajectory.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 8 16 32 \
  --rollout_segments 80 \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy \
  --planner_noop_risk_penalty 1.0 \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_energy_riskpen1.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = {
    "stage": "${STAGE}",
    "v64_edit_projection_checkpoint": "${V64_CKPT}",
    "v63_protected_reward_tau_checkpoint": "${V63_CKPT}",
    "merged_checkpoint": "${MERGED_CKPT}",
    "dataset_cache": "${CACHE}",
    "diagnostic": "v64 edit/projection branch with v63 reward/tau/risk head initialization; train reward+duration heads only",
    "files": {},
}
for name in [
    "metrics.json",
    "eval_time_alignment_final.json",
    "eval_long_trajectory.json",
    "eval_long_energy_riskpen1.json",
]:
    path = save_dir / name
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    item = {}
    if "val" in data:
        item["val"] = data["val"]
    if "dataset" in data:
        item["dataset"] = data["dataset"]
    if "noop_risk" in data:
        item["noop_risk"] = data["noop_risk"]
    for key in [
        "reward_sum",
        "tau_expected",
        "cumulative",
        "chosen_k_histogram",
        "completed_rollout_segments",
        "requested_rollout_segments",
        "stop_reason",
        "skipped_noop",
        "planner_score_mode",
        "planner_noop_risk_penalty",
    ]:
        if key in data:
            item[key] = data[key]
    summary["files"][name] = item
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
