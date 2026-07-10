#!/bin/bash
# Long-horizon (4000-tick) no-Perceiver pipeline, fold-0, fully chained.
#
#   1. Re-pretrain the JEPA encoder at window_size=4000 (conv strides 5,4,2,1
#      -> 98 patches), ckpt -> models/bl7_w4000/bl7_best.pt
#   2. Joint-train the cross-attn AR-modern decoder on the frozen 4000-tick
#      encoder's axial tokens (~686), encoding raw windows on the fly.
#   3. Joint 100K eval (encode-on-the-fly; no precomputed embeddings).
#
# Each stage afterok-chains the previous, so the whole thing can sit in the
# queue behind other jobs and run in order whenever GPU frees up.
#
# Usage:
#   bash submit_bl7_w4000_axial.sh                 # pretrain + fold-0 train + eval
#   PRETRAIN_DEP=<jobid> bash submit_bl7_w4000_axial.sh   # chain after an
#                                                  # already-queued pretrain
#   SKIP_PRETRAIN=1 bash submit_bl7_w4000_axial.sh  # ckpt already exists
#
# Folds 1-4 are intentionally NOT queued here; gate them on the fold-0 100K
# per-state NED (kill criteria: state-0 < 0.080, state-2 < 0.095, all < 0.046).

set -euo pipefail
SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

SEED="${SEED:-42}"
PARTITION="${PARTITION:-short}"
# ENC_CKPT / VARIANT env-overridable so the same chain can run a no-Perceiver
# decoder over a different 4000-tick encoder (e.g. the contrastive bl7_w4000_v2).
W4000_CKPT="${ENC_CKPT:-$SCRIPT_DIR/models/bl7_w4000/bl7_best.pt}"
VARIANT="${VARIANT:-ar_modern_bl7_w4000_axial}"
FOLD=0
FOLD_DIR="$SCRIPT_DIR/models/${VARIANT}_fold_${FOLD}"
RESULT_DIR="$SCRIPT_DIR/results/${VARIANT}_fold_${FOLD}"
mkdir -p "$FOLD_DIR" "$RESULT_DIR"

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
FOLD_TRAIN_0=("2025-03-17_10-36-44" "2025-03-20_09-31-56")

# ── Stage 1: encoder pretrain (4000-tick) ────────────────────────────────
PRETRAIN_JOB=""
if [ "${SKIP_PRETRAIN:-0}" != "1" ]; then
    DEP_ARG=""
    [ -n "${PRETRAIN_DEP:-}" ] && DEP_ARG="--dependency=afterok:${PRETRAIN_DEP}"
    PRETRAIN_JOB=$(sbatch --parsable $DEP_ARG \
        --gres=gpu:H200:1 --time=24:00:00 \
        --job-name="bl7_w4000_pretrain" \
        run_bl7_pretrain.sh \
        --window-size 4000 --conv-strides 5,4,2,1 \
        --mask-span-min 18 --mask-span-max 36 \
        --batch-size 16 --grad-accum 8 --max-steps 30000 \
        --ckpt-subdir bl7_w4000 --model-dir "$SCRIPT_DIR/models")
    echo "Stage 1 pretrain -> $PRETRAIN_JOB"
fi

# ── Stage 2: joint cross-attn decoder train (fold-0) ─────────────────────
DEP_ARG=""
[ -n "$PRETRAIN_JOB" ] && DEP_ARG="--dependency=afterok:${PRETRAIN_JOB}"
TRAIN_JOB=$(sbatch --parsable $DEP_ARG \
    --partition="$PARTITION" --gres=gpu:1 --time=12:00:00 \
    --exclude=gpu-6-[01-20] \
    --job-name="${VARIANT}_f${FOLD}_train" \
    --output="$FOLD_DIR/slurm_${VARIANT}_train_%j.log" \
    run_ar_modern_train.sh \
    --seed "$SEED" \
    --train-sessions "${FOLD_TRAIN_0[@]}" \
    --val-sessions "${VAL_SESSIONS[@]}" \
    --model-dir "$FOLD_DIR" \
    --max-steps 4000 --warmup-steps 500 --eval-every 500 --save-every 1000 \
    --bl7-encoder-ckpt "$W4000_CKPT" --bl7-axial --bl7-window-size 4000 \
    --use-cross-attn --d-sensor-emb 512 --n-sensor-tokens 686 --cond-dropout-p 0.10)
echo "Stage 2 joint train (fold 0) -> $TRAIN_JOB"

# ── Stage 3: joint 100K eval (fold-0) ────────────────────────────────────
EVAL_TRIGGER=$(sbatch --parsable \
    --dependency=afterok:${TRAIN_JOB} \
    --partition="$PARTITION" --cpus-per-task=2 --mem=4G --time=00:15:00 \
    --output="$RESULT_DIR/strat_trigger_%j.log" \
    --wrap="cd $SCRIPT_DIR && CHUNK_TIME=12:00:00 bash submit_stratified_100k.sh ${VARIANT} 0")
echo "Stage 3 100K eval trigger (fold 0) -> $EVAL_TRIGGER"

echo ""
echo "Chained: pretrain(${PRETRAIN_JOB:-skipped}) -> train($TRAIN_JOB) -> eval($EVAL_TRIGGER)"
echo "Monitor: squeue -u \$USER --format='%.10i %.32j %.8T %.10M %R'"
