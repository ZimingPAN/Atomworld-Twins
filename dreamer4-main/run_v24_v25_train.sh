#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# Remote commands for v22 extended eval, v24, v25 training
# Run these on the GPU server (likun@101.126.84.211)
# ═══════════════════════════════════════════════════════════

set -e
cd /home/likun/panziming/AtomWorld-Twins/dreamer4-main

# ─── Step 0: Copy v22 results to new baseline name ───
echo "=== Step 0: Rename v22 → macro_dreamer_kmc_baseline ==="
cp -r results/dreamer_macro_edit_v22_extreme results/macro_dreamer_kmc_baseline
echo "Done: results/macro_dreamer_kmc_baseline/"

# ─── Step 1: Extended eval on v22 (train split, 2000 samples) ───
echo "=== Step 1: v22 Extended Eval (train split) ==="
python eval_macro_time_alignment.py \
    --checkpoint results/macro_dreamer_kmc_baseline/best_model.pt \
    --cache results/dreamer_macro_edit_v15/segments.pt \
    --split train --save_all_samples \
    --output results/macro_dreamer_kmc_baseline/eval_train_split.json \
    2>&1 | tee results/macro_dreamer_kmc_baseline/eval_train_split.log
echo "Done: eval_train_split.json"

# ─── Step 2: Find Dreamer v9 checkpoint (for v25) ───
echo "=== Step 2: Locating Dreamer v9 checkpoint ==="
DREAMER_V9_CKPT=$(find /home/likun/panziming/AtomWorld-Twins -name 'best_model.pt' -path '*dreamer*v9*' 2>/dev/null | head -1)
if [ -z "$DREAMER_V9_CKPT" ]; then
    # Try broader search
    DREAMER_V9_CKPT=$(find /home/likun/panziming/AtomWorld-Twins -name 'best_model.pt' -path '*dreamer*' | grep -v macro | head -1)
fi
echo "Dreamer v9 checkpoint: $DREAMER_V9_CKPT"

# ─── Step 3: Train v24 (energy fix, resume from v22) ───
# GPU 0
echo "=== Step 3: Training v24 (energy fix) on GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 nohup python train_dreamer_macro_edit.py \
    --save_dir results/dreamer_macro_edit_v24_energy_fix \
    --dataset_cache results/dreamer_macro_edit_v15/segments.pt \
    --resume results/macro_dreamer_kmc_baseline/best_model.pt \
    --epochs 120 --reward_weight 3.0 \
    --proj_weight 10.0 --proj_l1_score_weight 5.0 \
    --device cuda \
    2>&1 | tee results/dreamer_macro_edit_v24_energy_fix/train.log &
V24_PID=$!
echo "v24 training started (PID=$V24_PID)"

# ─── Step 4: Train v25 (neural teacher, from scratch) ───
# GPU 1 (if available, otherwise wait for v24 to finish)
echo "=== Step 4: Training v25 (neural teacher) on GPU 1 ==="
if [ -n "$DREAMER_V9_CKPT" ]; then
    CUDA_VISIBLE_DEVICES=1 nohup python train_dreamer_macro_edit.py \
        --save_dir results/dreamer_macro_edit_v25_neural_teacher \
        --dataset_cache results/dreamer_macro_edit_v25_neural_teacher/segments.pt \
        --teacher_mode neural \
        --neural_teacher_path "$DREAMER_V9_CKPT" \
        --epochs 80 --reward_weight 2.0 \
        --proj_weight 10.0 --proj_l1_score_weight 5.0 \
        --device cuda \
        2>&1 | tee results/dreamer_macro_edit_v25_neural_teacher/train.log &
    V25_PID=$!
    echo "v25 training started (PID=$V25_PID)"
else
    echo "WARNING: Dreamer v9 checkpoint not found, skipping v25"
    echo "Please set DREAMER_V9_CKPT manually and re-run step 4"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Training jobs launched. Monitor with:"
echo "  tail -f results/dreamer_macro_edit_v24_energy_fix/train.log"
echo "  tail -f results/dreamer_macro_edit_v25_neural_teacher/train.log"
echo "═══════════════════════════════════════════════════════════"
