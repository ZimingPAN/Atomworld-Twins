#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RESULT_DIR=""
TRAIN_LOG=""
CACHE_PATH=""
DEVICE="cpu"
POLL_SECONDS=300
TITLE_NAME="AtomWorld-Twins"
MODEL_LABEL="AtomWorld-Twins"
PPO_LABEL="SwarmThinkers PPO"
PPO_EVAL="${ROOT_DIR}/results/ppo_v9_results/ppo_macro_eval_val.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LONG_ROLLOUT_SEGMENTS=500

usage() {
  cat <<'EOF'
Usage: bash fig/watch_macro_training_and_eval.sh --result-dir <dir> --train-log <log> --cache <segments.pt> [options]

Options:
  --result-dir <dir>            Result directory containing checkpoints and final_model.pt
  --train-log <path>            Training log file to monitor
  --cache <path>                Segment cache for paired evaluation
  --device <device>             Evaluation device, default: cpu
  --poll-seconds <n>            Poll interval in seconds, default: 300
  --python-bin <path>           Python executable, default: $PYTHON_BIN or python3
  --title-name <name>           Plot title, default: AtomWorld-Twins
  --model-label <name>          Plot label, default: AtomWorld-Twins
  --ppo-label <name>            PPO label, default: SwarmThinkers PPO
  --ppo-eval <path>             PPO paired eval json
  --long-rollout-segments <n>   Number of contiguous teacher-forced segments for final long eval, default: 500
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result-dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    --train-log)
      TRAIN_LOG="$2"
      shift 2
      ;;
    --cache)
      CACHE_PATH="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --title-name)
      TITLE_NAME="$2"
      shift 2
      ;;
    --model-label)
      MODEL_LABEL="$2"
      shift 2
      ;;
    --ppo-label)
      PPO_LABEL="$2"
      shift 2
      ;;
    --ppo-eval)
      PPO_EVAL="$2"
      shift 2
      ;;
    --long-rollout-segments)
      LONG_ROLLOUT_SEGMENTS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$RESULT_DIR" || -z "$TRAIN_LOG" || -z "$CACHE_PATH" ]]; then
  usage
  exit 1
fi

RESULT_DIR="$(cd "$RESULT_DIR" && pwd)"
TRAIN_LOG="$(cd "$(dirname "$TRAIN_LOG")" && pwd)/$(basename "$TRAIN_LOG")"
CACHE_PATH="$(cd "$(dirname "$CACHE_PATH")" && pwd)/$(basename "$CACHE_PATH")"

PROGRESS_LOG="${RESULT_DIR}/watch_progress.log"
PROGRESS_STATE="${RESULT_DIR}/.watch_progress_last"
FINAL_STATE="${RESULT_DIR}/.watch_final_eval_done"

snapshot_progress() {
  if [[ ! -f "$TRAIN_LOG" ]]; then
    return
  fi
  local snapshot
  snapshot="$(grep -E '\[Epoch|>>> VAL' "$TRAIN_LOG" | tail -n 4 || true)"
  if [[ -z "$snapshot" ]]; then
    return
  fi
  local previous=""
  if [[ -f "$PROGRESS_STATE" ]]; then
    previous="$(cat "$PROGRESS_STATE")"
  fi
  if [[ "$snapshot" != "$previous" ]]; then
    {
      printf '[%s]\n' "$(date '+%F %T')"
      printf '%s\n\n' "$snapshot"
    } >> "$PROGRESS_LOG"
    printf '%s' "$snapshot" > "$PROGRESS_STATE"
    echo "[watch] progress updated"
  fi
}

run_final_eval() {
  local checkpoint_path="${RESULT_DIR}/final_model.pt"
  local paired_json="${RESULT_DIR}/eval_time_alignment_realized_final.json"
  local paired_fig="${RESULT_DIR}/macro_edit_eval_comparison_final.png"
  local time_fig="${RESULT_DIR}/macro_edit_time_alignment_final.png"
  local realized_fig="${RESULT_DIR}/macro_edit_realized_time_final.png"
  local long_json="${RESULT_DIR}/eval_long_trajectory_${LONG_ROLLOUT_SEGMENTS}.json"

  echo "[watch] final_model.pt detected, running final evaluations"
  "${PYTHON_BIN}" "${ROOT_DIR}/fig/run_realized_eval_and_plots.py" \
    --checkpoint "$checkpoint_path" \
    --cache "$CACHE_PATH" \
    --model-dir "$RESULT_DIR" \
    --device "$DEVICE" \
    --title-name "$TITLE_NAME" \
    --model-label "$MODEL_LABEL" \
    --ppo-label "$PPO_LABEL" \
    --ppo-eval "$PPO_EVAL" \
    --eval-output "$paired_json" \
    --comparison-eval-output "$paired_fig" \
    --comparison-time-output "$time_fig" \
    --realized-output "$realized_fig"

  "${PYTHON_BIN}" "${ROOT_DIR}/dreamer4-main/eval_macro_long_trajectory.py" \
    --checkpoint "$checkpoint_path" \
    --device "$DEVICE" \
    --rollout_segments "$LONG_ROLLOUT_SEGMENTS" \
    --output "$long_json"

  touch "$FINAL_STATE"
  echo "[watch] final evaluations completed"
}

while true; do
  snapshot_progress
  if [[ -f "${RESULT_DIR}/final_model.pt" ]]; then
    if [[ ! -f "$FINAL_STATE" ]]; then
      run_final_eval
    fi
    echo "[watch] done"
    break
  fi
  sleep "$POLL_SECONDS"
done