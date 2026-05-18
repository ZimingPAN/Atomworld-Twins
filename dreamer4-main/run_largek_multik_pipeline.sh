#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
BASE_CKPT="${BASE_CKPT:-results/kmc_teacher_dreamer_macro_wm/final_model.pt}"
RESULT_ROOT="${RESULT_ROOT:-results/largek_multik_$(date +%m%d_%H%M%S)}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-4096}"
MAX_CANDIDATE_SITES="${MAX_CANDIDATE_SITES:-1024}"
MAX_SEGMENTS_PER_ROLLOUT="${MAX_SEGMENTS_PER_ROLLOUT:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PROJ_EVERY_N_BATCHES="${PROJ_EVERY_N_BATCHES:-2}"
COUNT_LOSS_WEIGHT="${COUNT_LOSS_WEIGHT:-0.1}"
PAIR_WEIGHT="${PAIR_WEIGHT:-0.0}"
PRIOR_PAIR_WEIGHT="${PRIOR_PAIR_WEIGHT:-${PAIR_WEIGHT}}"
SMOKE="${SMOKE:-0}"
START_STAGE="${START_STAGE:-v53_largek_16_32_64_128}"
START_STAGE_INIT_CKPT="${START_STAGE_INIT_CKPT:-}"
PLANNER_SELECTED_DATA="${PLANNER_SELECTED_DATA:-0}"
PLANNER_SELECTED_MIN_PROJECTED_CHANGED_SITES="${PLANNER_SELECTED_MIN_PROJECTED_CHANGED_SITES:-0}"
PLANNER_SELECTED_DURATION_SOURCE="${PLANNER_SELECTED_DURATION_SOURCE:-model}"
PLANNER_SELECTED_TAU_SOURCE="${PLANNER_SELECTED_TAU_SOURCE:-model}"
PLANNER_SELECTED_SCORE_MODE="${PLANNER_SELECTED_SCORE_MODE:-energy_per_sqrt_tau}"
PLANNER_SELECTED_REWARD_SOURCE="${PLANNER_SELECTED_REWARD_SOURCE:-projected}"
LONG_MIN_PROJECTED_CHANGED_SITES="${LONG_MIN_PROJECTED_CHANGED_SITES:-2}"
NATURAL_TEACHER_BACKEND="${NATURAL_TEACHER_BACKEND:-kmc}"
SEGMENT_BOUNDARY_MODE="${SEGMENT_BOUNDARY_MODE:-fixed_k}"
ADAPTIVE_MIN_K="${ADAPTIVE_MIN_K:-1}"
ADAPTIVE_CANDIDATE_HORIZON_SOURCE="${ADAPTIVE_CANDIDATE_HORIZON_SOURCE:-actual}"
ADAPTIVE_KEY_MOVING_TYPES="${ADAPTIVE_KEY_MOVING_TYPES:-1}"
ADAPTIVE_MIN_TOUCHED_SITES="${ADAPTIVE_MIN_TOUCHED_SITES:-0}"
ADAPTIVE_ABS_DELTA_E_THRESHOLD="${ADAPTIVE_ABS_DELTA_E_THRESHOLD:-0.0}"
ADAPTIVE_CUMULATIVE_ABS_DELTA_E_THRESHOLD="${ADAPTIVE_CUMULATIVE_ABS_DELTA_E_THRESHOLD:-0.0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Missing BASE_CKPT=${BASE_CKPT}" >&2
  exit 2
fi

mkdir -p "${RESULT_ROOT}"

if [[ "${SMOKE}" == "1" ]]; then
  TRAIN_PER_K="${TRAIN_PER_K:-12}"
  VAL_PER_K="${VAL_PER_K:-4}"
  EPOCHS="${EPOCHS:-1}"
  LONG_SEGMENTS="${LONG_SEGMENTS:-8}"
else
  TRAIN_PER_K="${TRAIN_PER_K:-128}"
  VAL_PER_K="${VAL_PER_K:-32}"
  EPOCHS="${EPOCHS:-20}"
  LONG_SEGMENTS="${LONG_SEGMENTS:-80}"
fi

run_stage() {
  local stage="$1"
  local init_ckpt="$2"
  shift 2
  local horizons=("$@")
  local save_dir="${RESULT_ROOT}/${stage}"
  local cache="${save_dir}/segments.pt"
  local log="${save_dir}/pipeline.log"
  mkdir -p "${save_dir}"

  {
    echo "=== ${stage} START $(date -Is) ==="
    echo "init=${init_ckpt}"
    echo "horizons=${horizons[*]}"
    echo "train_per_k=${TRAIN_PER_K} val_per_k=${VAL_PER_K} epochs=${EPOCHS}"
    echo "max_episode_steps=${MAX_EPISODE_STEPS} max_candidate_sites=${MAX_CANDIDATE_SITES} max_segments_per_rollout=${MAX_SEGMENTS_PER_ROLLOUT}"
    echo "planner_selected_data=${PLANNER_SELECTED_DATA} planner_min_projected_changed=${PLANNER_SELECTED_MIN_PROJECTED_CHANGED_SITES}"
    echo "natural_teacher_backend=${NATURAL_TEACHER_BACKEND} segment_boundary_mode=${SEGMENT_BOUNDARY_MODE} adaptive_min_k=${ADAPTIVE_MIN_K}"
  } | tee -a "${log}"

  local planner_selected_args=()
  if [[ "${PLANNER_SELECTED_DATA}" == "1" ]]; then
    planner_selected_args=(
      --planner_selected_from "${init_ckpt}"
      --planner_selected_min_projected_changed_sites "${PLANNER_SELECTED_MIN_PROJECTED_CHANGED_SITES}"
      --planner_selected_duration_source "${PLANNER_SELECTED_DURATION_SOURCE}"
      --planner_selected_tau_source "${PLANNER_SELECTED_TAU_SOURCE}"
      --planner_selected_score_mode "${PLANNER_SELECTED_SCORE_MODE}"
      --planner_selected_reward_prediction_source "${PLANNER_SELECTED_REWARD_SOURCE}"
    )
  fi

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_dreamer_macro_edit.py \
    --save_dir "${save_dir}" \
    --dataset_cache "${cache}" \
    --init_from "${init_ckpt}" \
    --segment_ks "${horizons[@]}" \
    --train_segments_per_k "${TRAIN_PER_K}" \
    --val_segments_per_k "${VAL_PER_K}" \
    --max_episode_steps "${MAX_EPISODE_STEPS}" \
    --max_segments_per_rollout "${MAX_SEGMENTS_PER_ROLLOUT}" \
    --max_candidate_sites "${MAX_CANDIDATE_SITES}" \
    --natural_teacher_backend "${NATURAL_TEACHER_BACKEND}" \
    --segment_boundary_mode "${SEGMENT_BOUNDARY_MODE}" \
    --adaptive_min_k "${ADAPTIVE_MIN_K}" \
    --adaptive_candidate_horizon_source "${ADAPTIVE_CANDIDATE_HORIZON_SOURCE}" \
    --adaptive_key_moving_types ${ADAPTIVE_KEY_MOVING_TYPES} \
    --adaptive_min_touched_sites "${ADAPTIVE_MIN_TOUCHED_SITES}" \
    --adaptive_abs_delta_e_threshold "${ADAPTIVE_ABS_DELTA_E_THRESHOLD}" \
    --adaptive_cumulative_abs_delta_e_threshold "${ADAPTIVE_CUMULATIVE_ABS_DELTA_E_THRESHOLD}" \
    --teacher_path_summary_mode stepwise \
    --tau_supervision_mode prior_main \
    --reward_prediction_source projected \
    --realized_tau_weight 0.1 \
    --tau_log_mu_weight 1.0 \
    --count_loss_weight "${COUNT_LOSS_WEIGHT}" \
    --pair_weight "${PAIR_WEIGHT}" \
    --prior_pair_weight "${PRIOR_PAIR_WEIGHT}" \
    --proj_every_n_batches "${PROJ_EVERY_N_BATCHES}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --device "${DEVICE}" \
    "${planner_selected_args[@]}" \
    2>&1 | tee -a "${log}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_time_alignment.py \
    --checkpoint "${save_dir}/final_model.pt" \
    --cache "${cache}" \
    --split val \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --output "${save_dir}/eval_time_alignment_final.json" \
    --save_all_samples \
    2>&1 | tee -a "${log}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
    --checkpoint "${save_dir}/final_model.pt" \
    --planner_segment_ks "${horizons[@]}" \
    --rollout_segments "${LONG_SEGMENTS}" \
    --max_episode_steps_override "${MAX_EPISODE_STEPS}" \
    --duration_source model \
    --planner_tau_source model \
    --planner_score_mode energy_per_sqrt_tau \
    --min_projected_changed_sites "${LONG_MIN_PROJECTED_CHANGED_SITES}" \
    --device "${DEVICE}" \
    --output "${save_dir}/eval_long_trajectory.json" \
    2>&1 | tee -a "${log}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_macro_long_trajectory.py \
    --checkpoint "${save_dir}/final_model.pt" \
    --planner_segment_ks "${horizons[@]}" \
    --rollout_segments "${LONG_SEGMENTS}" \
    --max_episode_steps_override "${MAX_EPISODE_STEPS}" \
    --duration_source baseline \
    --planner_tau_source baseline \
    --planner_score_mode energy_per_sqrt_tau \
    --min_projected_changed_sites "${LONG_MIN_PROJECTED_CHANGED_SITES}" \
    --device "${DEVICE}" \
    --output "${save_dir}/eval_long_trajectory_baselinetau.json" \
    2>&1 | tee -a "${log}"

  "${PYTHON_BIN}" - "${save_dir}" <<'PY' | tee -a "${log}"
import json
import sys
from pathlib import Path

save_dir = Path(sys.argv[1])
paired_path = save_dir / "eval_time_alignment_final.json"
long_path = save_dir / "eval_long_trajectory.json"
summary = {"stage": save_dir.name}
if paired_path.exists():
    paired = json.loads(paired_path.read_text())
    summary["paired_tau_log_mae"] = paired.get("tau_expected", {}).get("log_mae")
    summary["paired_tau_log_corr"] = paired.get("tau_expected", {}).get("log_corr")
    summary["paired_tau_scale_ratio"] = paired.get("tau_expected", {}).get("scale_ratio")
    summary["paired_change_f1"] = paired.get("edit", {}).get("change_f1")
    summary["paired_projected_changed_type_acc"] = paired.get("edit", {}).get("projected_changed_type_acc")
    summary["paired_reachability_violation_rate"] = paired.get("projection", {}).get("reachability_violation_rate")
if long_path.exists():
    long = json.loads(long_path.read_text())
    summary["long_completed"] = long.get("completed_rollout_segments")
    summary["long_stop_reason"] = long.get("stop_reason")
    summary["long_tau_log_mae"] = long.get("tau_expected", {}).get("log_mae")
    summary["long_tau_scale_ratio"] = long.get("tau_expected", {}).get("scale_ratio")
    summary["long_expected_time_ratio"] = long.get("cumulative", {}).get("expected_time_ratio")
(save_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps({"stage_summary": summary}, sort_keys=True))
PY

  echo "=== ${stage} DONE $(date -Is) ===" | tee -a "${log}"
}

cd "$(dirname "$0")"

run_from_start=0
maybe_run_stage() {
  local stage="$1"
  if [[ "${stage}" == "${START_STAGE}" ]]; then
    run_from_start=1
    if [[ -n "${START_STAGE_INIT_CKPT}" ]]; then
      shift
      shift
      run_stage "${stage}" "${START_STAGE_INIT_CKPT}" "$@"
      return
    fi
  fi
  if [[ "${run_from_start}" == "1" ]]; then
    run_stage "$@"
  else
    echo "=== ${stage} SKIP before START_STAGE=${START_STAGE} ==="
  fi
}

maybe_run_stage "v53_largek_16_32_64_128" "${BASE_CKPT}" 16 32 64 128
maybe_run_stage "v54_largek_32_64_128_256" "${RESULT_ROOT}/v53_largek_16_32_64_128/final_model.pt" 32 64 128 256
maybe_run_stage "v55_largek_64_128_256_512" "${RESULT_ROOT}/v54_largek_32_64_128_256/final_model.pt" 64 128 256 512
maybe_run_stage "v56_largek_128_256_512_1024" "${RESULT_ROOT}/v55_largek_64_128_256_512/final_model.pt" 128 256 512 1024

echo "Large-k pipeline complete: ${RESULT_ROOT}"
