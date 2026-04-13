#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT_DIR=""
CACHE_PATH=""
DEVICE="cpu"
POLL_SECONDS=60
TAG=""
TITLE_NAME="AtomWorld-Twins"
MODEL_LABEL="AtomWorld-Twins"
PPO_LABEL="SwarmThinkers PPO"
PPO_EVAL="${ROOT_DIR}/results/ppo_v9_results/ppo_macro_eval_val.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ONCE=0
EXIT_ON_FINAL=0

usage() {
  cat <<'EOF'
Usage: bash fig/run_realized_eval_watch.sh --checkpoint-dir <dir> --cache <segments.pt> [options]

Options:
  --checkpoint-dir <dir>   Directory containing checkpoint_*.pt / best_model.pt / final_model.pt
  --cache <path>           Segment cache path used for evaluation
  --device <device>        Evaluation device, default: cpu
  --poll-seconds <n>       Poll interval in seconds for watch mode, default: 60
  --tag <name>             Output suffix tag, default: basename(checkpoint-dir)
  --title-name <name>      Plot title name, default: AtomWorld-Twins
  --model-label <name>     Plot model label, default: AtomWorld-Twins
  --ppo-label <name>       PPO label, default: SwarmThinkers PPO
  --ppo-eval <path>        PPO paired-eval json, default: results/ppo_v9_results/ppo_macro_eval_val.json
  --python-bin <path>      Python executable, default: $PYTHON_BIN or python3
  --once                   Evaluate the newest checkpoint once and exit
  --exit-on-final          In watch mode, exit after final_model.pt has been evaluated
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
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
    --tag)
      TAG="$2"
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
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --once)
      ONCE=1
      shift
      ;;
    --exit-on-final)
      EXIT_ON_FINAL=1
      shift
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

if [[ -z "$CHECKPOINT_DIR" || -z "$CACHE_PATH" ]]; then
  usage
  exit 1
fi

CHECKPOINT_DIR="$(cd "$CHECKPOINT_DIR" && pwd)"
CACHE_PATH="$(cd "$(dirname "$CACHE_PATH")" && pwd)/$(basename "$CACHE_PATH")"
TAG="${TAG:-$(basename "$CHECKPOINT_DIR")}"
STATE_FILE="${CHECKPOINT_DIR}/.realized_eval_last_checkpoint"

find_latest_checkpoint() {
  python3 - <<'PY' "$CHECKPOINT_DIR"
from pathlib import Path
import sys

checkpoint_dir = Path(sys.argv[1])
candidates = []
for pattern in ("final_model.pt", "best_model.pt", "checkpoint_*.pt"):
    candidates.extend(checkpoint_dir.glob(pattern))
candidates = [path for path in candidates if path.is_file()]
if not candidates:
    sys.exit(0)
latest = max(candidates, key=lambda path: (path.stat().st_mtime, path.name))
print(latest)
PY
}

run_eval_for_checkpoint() {
  local checkpoint_path="$1"
  local checkpoint_name
  local eval_output
  local comparison_eval_output
  local comparison_time_output
  local realized_output

  checkpoint_name="$(basename "$checkpoint_path" .pt)"
  eval_output="${CHECKPOINT_DIR}/eval_time_alignment_realized_${checkpoint_name}.json"
  comparison_eval_output="${ROOT_DIR}/fig/macro_edit_eval_comparison_${TAG}_${checkpoint_name}.png"
  comparison_time_output="${ROOT_DIR}/fig/macro_edit_time_alignment_${TAG}_${checkpoint_name}.png"
  realized_output="${ROOT_DIR}/fig/macro_edit_realized_time_${TAG}_${checkpoint_name}.png"

  echo "[watch] evaluating ${checkpoint_path}"
  "${PYTHON_BIN}" "${ROOT_DIR}/fig/run_realized_eval_and_plots.py" \
    --checkpoint "${checkpoint_path}" \
    --cache "${CACHE_PATH}" \
    --model-dir "${CHECKPOINT_DIR}" \
    --device "${DEVICE}" \
    --title-name "${TITLE_NAME}" \
    --model-label "${MODEL_LABEL}" \
    --ppo-label "${PPO_LABEL}" \
    --ppo-eval "${PPO_EVAL}" \
    --eval-output "${eval_output}" \
    --comparison-eval-output "${comparison_eval_output}" \
    --comparison-time-output "${comparison_time_output}" \
    --realized-output "${realized_output}"
  printf '%s\n' "${checkpoint_path}" > "${STATE_FILE}"
}

while true; do
  latest_checkpoint="$(find_latest_checkpoint || true)"
  if [[ -n "${latest_checkpoint}" ]]; then
    last_checkpoint=""
    if [[ -f "${STATE_FILE}" ]]; then
      last_checkpoint="$(cat "${STATE_FILE}")"
    fi
    if [[ "${latest_checkpoint}" != "${last_checkpoint}" ]]; then
      run_eval_for_checkpoint "${latest_checkpoint}"
      if [[ ${ONCE} -eq 1 ]]; then
        break
      fi
      if [[ ${EXIT_ON_FINAL} -eq 1 && "$(basename "${latest_checkpoint}")" == "final_model.pt" ]]; then
        echo "[watch] final_model.pt evaluated, exiting"
        break
      fi
    elif [[ ${ONCE} -eq 1 ]]; then
      echo "[watch] latest checkpoint already evaluated: ${latest_checkpoint}"
      break
    fi
  else
    echo "[watch] no checkpoint found yet in ${CHECKPOINT_DIR}"
    if [[ ${ONCE} -eq 1 ]]; then
      break
    fi
  fi

  if [[ ${ONCE} -eq 1 ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done