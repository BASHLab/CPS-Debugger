#!/bin/bash
# Step 2 of the 4000-tick ablation: coupling K-sweep on E* = step-1000
# (models/bl7_w4000/bl7_best.pt). Joint-train a fresh K-latent Perceiver pool
# + cross-attn AR-modern decoder for each K, then extract encoder -> precompute
# (n,K,256) embeddings -> 100K stratified eval. Fold-0 only.
#
# Endpoints already known (no re-run):
#   prefix-K4 (mean-pool)  ar_modern_bl7_w4000_prefix_s1000  agg 0.0725
#   no-Perceiver (~686)    ar_modern_bl7_w4000_axial         agg 0.0407  (= R4)
# This sweep fills the cross-attn pooled middle: K = 4 8 16 32 64.
#
# Usage:
#   bash submit_bl7_w4000_ksweep.sh                 # K = 4 8 16 32 64
#   K_VALUES="16 32" bash submit_bl7_w4000_ksweep.sh

set -euo pipefail
SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

K_VALUES="${K_VALUES:-4 8 16 32 64}"
SEED="${SEED:-42}"
PARTITION="${PARTITION:-short}"
# ENC_CKPT / VARIANT_PREFIX env-overridable so the same K-sweep can run over a
# different 4000-tick encoder (e.g. the contrastive bl7_w4000_v2).
ENC_CKPT="${ENC_CKPT:-$SCRIPT_DIR/models/bl7_w4000/bl7_best.pt}"     # default E* = step-1000
VARIANT_PREFIX="${VARIANT_PREFIX:-ar_modern_bl7_w4000}"
test -f "$ENC_CKPT" || { echo "ERROR: $ENC_CKPT missing" >&2; exit 1; }

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")
FOLD_TRAIN_0=("2025-03-17_10-36-44" "2025-03-20_09-31-56")

for K in $K_VALUES; do
    VARIANT="${VARIANT_PREFIX}_k${K}"
    FOLD_DIR="$SCRIPT_DIR/models/${VARIANT}_fold_0"
    RESULT_DIR="$SCRIPT_DIR/results/${VARIANT}_fold_0"
    EMB_DIR="$SCRIPT_DIR/sensor_embeddings_${VARIANT}_fold_0"
    mkdir -p "$FOLD_DIR" "$RESULT_DIR" "$EMB_DIR"

    echo "── K=$K → $VARIANT ──"

    # ── Stage 1: joint train (frozen backbone + fresh K-pool + cross-attn) ──
    TRAIN_JOB=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name="${VARIANT}_train" \
        --output="$FOLD_DIR/slurm_${VARIANT}_train_%j.log" \
        run_ar_modern_train.sh \
        --seed "$SEED" \
        --train-sessions "${FOLD_TRAIN_0[@]}" \
        --val-sessions "${VAL_SESSIONS[@]}" \
        --model-dir "$FOLD_DIR" \
        --max-steps 4000 --warmup-steps 500 --eval-every 500 --save-every 1000 \
        --bl7-encoder-ckpt "$ENC_CKPT" --bl7-K "$K" --bl7-window-size 4000 \
        --use-cross-attn --d-sensor-emb 256 --n-sensor-tokens "$K" --cond-dropout-p 0.10)
    echo "  Stage 1 joint train -> $TRAIN_JOB"

    # ── Stage 2: extract trained encoder+pool, precompute test embeddings ──
    PP_JOB=$(sbatch --parsable \
        --dependency=afterok:${TRAIN_JOB} \
        --partition="$PARTITION" --gres=gpu:1 \
        --cpus-per-task=4 --mem=64G --time=02:00:00 \
        --exclude=gpu-6-[01-20] \
        --output="$FOLD_DIR/slurm_postprocess_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
python3 -u extract_bl7_encoder_from_joint.py \
    --joint-ckpt $FOLD_DIR/ar_modern_best.pt \
    --output $FOLD_DIR/bl7_encoder.pt \
    --orig-encoder-ckpt $ENC_CKPT
python3 -u sensor_encoder_bl7.py \
    --ckpt $FOLD_DIR/bl7_encoder.pt \
    --sessions ${TEST_SESSIONS[*]} \
    --output-dir $EMB_DIR \
    --batch-size 64 --device cuda")
    echo "  Stage 2 postprocess -> $PP_JOB (after $TRAIN_JOB)"

    # ── Stage 3: 100K stratified eval (fold-0, --emb-dir) ──────────────────
    EVAL_TRIGGER=$(sbatch --parsable \
        --dependency=afterok:${PP_JOB} \
        --partition="$PARTITION" --cpus-per-task=2 --mem=4G --time=00:15:00 \
        --output="$RESULT_DIR/strat_trigger_%j.log" \
        --wrap="cd $SCRIPT_DIR && bash submit_stratified_100k.sh ${VARIANT} 0")
    echo "  Stage 3 eval trig   -> $EVAL_TRIGGER (after $PP_JOB)"
    echo ""
done

echo "4000-tick K-sweep submitted. Monitor: squeue -u \$USER --format='%.10i %.34j %.8T %.10M %R'"
