#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-2}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_editrecover_0516}"
STAGE="${STAGE:-v61_rewardcal_editrecover_balanced}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
INIT_CKPT="${INIT_CKPT:-results/natural_teacher_support32_rewardcal_0516/v60_noopkeep_rewardcal/final_model.pt}"
PLANNER_CKPT="${PLANNER_CKPT:-results/natural_teacher_support32_noopsupport_0515/v58_adaptive2048_noop_support/final_model.pt}"
CACHE="${CACHE:-results/natural_teacher_support32_noopkeep_0515/v59_adaptive2048_noopkeep_hardneg/segments.pt}"
LOG="${SAVE_DIR}/pipeline.log"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "init=${INIT_CKPT}"
  echo "planner_signature=${PLANNER_CKPT}"
  echo "cache=${CACHE}"
  echo "fix=low-lr edit/projection recovery from rewardcal, with duration frozen and no-op reward guard retained"
} | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_dreamer_macro_edit.py \
  --save_dir "${SAVE_DIR}" \
  --dataset_cache "${CACHE}" \
  --init_from "${INIT_CKPT}" \
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
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --reward_prediction_source projected \
  --include_noop_segments \
  --keep_after_noop_segments \
  --freeze_duration_heads \
  --realized_tau_weight 0.05 \
  --tau_log_mu_weight 0.5 \
  --count_loss_weight 1.0 \
  --mask_sparsity_weight 0.01 \
  --pair_weight 0.25 \
  --prior_pair_weight 0.25 \
  --reward_weight 1.2 \
  --prior_reward_weight 1.2 \
  --reward_gate_weight 4.0 \
  --reward_zero_weight 8.0 \
  --reward_sign_weight 0.5 \
  --reward_magnitude_weight 1.0 \
  --prior_edit_weight 0.9 \
  --prior_latent_weight 0.3 \
  --proj_weight 1.1 \
  --no_aux_anneal \
  --natural_teacher_backend kmc \
  --segment_boundary_mode adaptive_key_event \
  --adaptive_min_k 8 \
  --adaptive_candidate_horizon_source actual \
  --adaptive_key_moving_types 1 \
  --disable_teacher_candidate_augmentation \
  --lr 1e-5 \
  --batch_size 2 \
  --epochs 4 \
  --device "${DEVICE}" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_time_alignment.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --cache "${CACHE}" \
  --split val \
  --batch_size 2 \
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
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_energy.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 8 16 32 \
  --rollout_segments 80 \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy_per_sqrt_tau \
  --planner_k_penalty_power 2.0 \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_kpen2.json" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --planner_segment_ks 8 16 32 \
  --rollout_segments 20 \
  --max_episode_steps_override 1024 \
  --duration_source model \
  --planner_tau_source model \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 0 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_min0_diagnostic.json" \
  2>&1 | tee -a "${LOG}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

save_dir = Path("${SAVE_DIR}")
summary = {
    "stage": "${STAGE}",
    "init_checkpoint": "${INIT_CKPT}",
    "planner_signature_checkpoint": "${PLANNER_CKPT}",
    "dataset_cache": "${CACHE}",
    "segment_ks": [8, 16, 32],
    "boundary": "adaptive_key_event",
    "support": "noopkeep_cache_rewardcal_edit_projection_recovery",
    "max_candidate_sites": 2048,
    "files": {},
}
for name in [
    "metrics.json",
    "eval_time_alignment_final.json",
    "eval_long_trajectory.json",
    "eval_long_energy.json",
    "eval_long_kpen2.json",
    "eval_long_min0_diagnostic.json",
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
        "planner_k_penalty_power",
    ]:
        if key in data:
            item[key] = data[key]
    summary["files"][name] = item
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
