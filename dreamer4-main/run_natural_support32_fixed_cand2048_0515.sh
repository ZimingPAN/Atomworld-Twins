#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
GPU_ID="${GPU_ID:-1}"
DEVICE="${DEVICE:-cuda}"
RESULT_ROOT="${RESULT_ROOT:-results/natural_teacher_support32_fixed_cand2048_0515}"
STAGE="${STAGE:-v58_fixed_support32_cand2048_noaug}"
SAVE_DIR="${RESULT_ROOT}/${STAGE}"
LOG="${SAVE_DIR}/pipeline.log"
INIT_CKPT="${INIT_CKPT:-results/largek_multik_0514/v53_largek_16_32_64_128/final_model.pt}"

mkdir -p "${SAVE_DIR}"

{
  echo "=== ${STAGE} START $(date -Is) ==="
  echo "init=${INIT_CKPT}"
  echo "segment_ks=8,16,32 boundary=fixed_k support=no_teacher_path_aug max_seed_vacancies=32 max_candidate_sites=2048"
} | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_dreamer_macro_edit.py \
  --save_dir "${SAVE_DIR}" \
  --dataset_cache "${SAVE_DIR}/segments.pt" \
  --init_from "${INIT_CKPT}" \
  --segment_ks 8 16 32 \
  --train_segments_per_k 192 \
  --val_segments_per_k 48 \
  --max_episode_steps 1024 \
  --max_segments_per_rollout 8 \
  --max_seed_vacancies 32 \
  --max_candidate_sites 2048 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --reward_prediction_source projected \
  --realized_tau_weight 0.1 \
  --tau_log_mu_weight 1.0 \
  --count_loss_weight 0.3 \
  --pair_weight 0.2 \
  --prior_pair_weight 0.2 \
  --prior_edit_weight 0.6 \
  --proj_weight 0.8 \
  --no_aux_anneal \
  --natural_teacher_backend kmc \
  --segment_boundary_mode fixed_k \
  --disable_teacher_candidate_augmentation \
  --batch_size 1 \
  --epochs 16 \
  --device "${DEVICE}" \
  2>&1 | tee -a "${LOG}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_time_alignment.py \
  --checkpoint "${SAVE_DIR}/final_model.pt" \
  --cache "${SAVE_DIR}/segments.pt" \
  --split val \
  --batch_size 1 \
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
  --duration_source baseline \
  --planner_tau_source baseline \
  --planner_score_mode energy_per_sqrt_tau \
  --min_projected_changed_sites 2 \
  --device "${DEVICE}" \
  --output "${SAVE_DIR}/eval_long_trajectory_baselinetau.json" \
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
    "segment_ks": [8, 16, 32],
    "boundary": "fixed_k",
    "support": "no_teacher_path_aug",
    "max_candidate_sites": 2048,
    "files": {},
}
for name in [
    "metrics.json",
    "eval_time_alignment_final.json",
    "eval_long_trajectory.json",
    "eval_long_trajectory_baselinetau.json",
    "eval_long_min0_diagnostic.json",
]:
    path = save_dir / name
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    item = {
        key: data.get(key)
        for key in [
            "num_samples",
            "completed_rollout_segments",
            "requested_rollout_segments",
            "stop_reason",
            "skipped_noop",
            "duration_source",
            "planner_tau_source",
        ]
    }
    if "reward_sum" in data:
        item["reward_sum"] = data.get("reward_sum")
    if "tau_expected" in data:
        item["tau_expected"] = data.get("tau_expected")
    if "val" in data:
        val = data["val"]
        item["val"] = {
            key: val.get(key)
            for key in [
                "reward_corr",
                "tau_log_mae",
                "tau_log_corr",
                "tau_scale_ratio",
                "change_f1",
                "projected_change_f1",
                "projected_changed_type_acc",
                "reachability_violation_rate",
            ]
        }
    summary["files"][name] = item
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f"Saved summary to {save_dir / 'stage_summary.json'}")
PY

echo "=== ${STAGE} DONE $(date -Is) ===" | tee -a "${LOG}"
