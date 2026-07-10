#!/bin/bash
# Step 1 of the 4000-tick ablation: pick the best encoder checkpoint with a
# cheap prefix-K4 probe. For each pretrain step, precompute encode_dense
# (n,4,256) embeddings from that encoder, train a prefix AR-modern decoder
# (fold-0), then run the 100K stratified eval. Pick E* by per-state NED
# (state-0 first, then aggregate).
#
# All cells are fold-0, decoder-only, off the on-disk 4000-tick encoders
# (models/bl7_w4000/). No new encoder pretraining.
#
# Usage:
#   bash submit_bl7_w4000_encsweep.sh                 # steps 1000 10000 20000 40000
#   STEPS="20000 40000" bash submit_bl7_w4000_encsweep.sh
#
# step 1000 maps to bl7_best.pt (= the step-1000 best-val ckpt; there is no
# bl7_step_1000.pt). All other steps map to bl7_step_<N>.pt.

set -euo pipefail
SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

STEPS="${STEPS:-1000 10000 20000 40000}"
SEED="${SEED:-42}"
PARTITION="${PARTITION:-short}"
ENC_DIR="$SCRIPT_DIR/models/bl7_w4000"

# fold-0 sessions: train ∪ val ∪ test (6 sessions total)
VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")
FOLD_TRAIN_0=("2025-03-17_10-36-44" "2025-03-20_09-31-56")
ALL_SESSIONS=("${FOLD_TRAIN_0[@]}" "${VAL_SESSIONS[@]}" "${TEST_SESSIONS[@]}")

for STEP in $STEPS; do
    if [ "$STEP" == "1000" ]; then
        CKPT="$ENC_DIR/bl7_best.pt"
    else
        CKPT="$ENC_DIR/bl7_step_${STEP}.pt"
    fi
    test -f "$CKPT" || { echo "ERROR: encoder ckpt $CKPT missing" >&2; exit 1; }

    VARIANT="ar_modern_bl7_w4000_prefix_s${STEP}"
    EMB_DIR="$SCRIPT_DIR/sensor_embeddings_${VARIANT}_fold_0"
    FOLD_DIR="$SCRIPT_DIR/models/${VARIANT}_fold_0"
    RESULT_DIR="$SCRIPT_DIR/results/${VARIANT}_fold_0"
    mkdir -p "$EMB_DIR" "$FOLD_DIR" "$RESULT_DIR"

    echo "── step $STEP → $VARIANT (ckpt $(basename "$CKPT")) ──"

    # ── Stage 1: precompute prefix-K4 embeddings (encode_dense → (n,4,256)) ──
    PRE_JOB=$(sbatch --parsable \
        --partition="$PARTITION" --gres=gpu:1 \
        --cpus-per-task=4 --mem=32G --time=12:00:00 \
        --exclude=gpu-6-[01-20] \
        --job-name="${VARIANT}_precompute" \
        --output="$EMB_DIR/slurm_precompute_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
python3 -u sensor_encoder_bl7.py \
    --ckpt $CKPT \
    --sessions ${ALL_SESSIONS[*]} \
    --output-dir $EMB_DIR \
    --batch-size 128 --device cuda")
    echo "  Stage 1 precompute -> $PRE_JOB"

    # ── Stage 2: prefix decoder train (fold-0, mean-pooled K=4 → 256) ────────
    TRAIN_JOB=$(sbatch --parsable \
        --dependency=afterok:${PRE_JOB} \
        --partition="$PARTITION" \
        --job-name="${VARIANT}_train" \
        --output="$FOLD_DIR/slurm_${VARIANT}_train_%j.log" \
        run_ar_modern_train.sh \
        --seed "$SEED" \
        --train-sessions "${FOLD_TRAIN_0[@]}" \
        --val-sessions "${VAL_SESSIONS[@]}" \
        --emb-dir "$EMB_DIR" \
        --d-sensor-emb 256 \
        --model-dir "$FOLD_DIR" \
        --max-steps 4000 --warmup-steps 500 --eval-every 500 --save-every 1000)
    echo "  Stage 2 train      -> $TRAIN_JOB (after $PRE_JOB)"

    # ── Stage 3: 100K stratified eval (fold-0) ───────────────────────────────
    EVAL_TRIGGER=$(sbatch --parsable \
        --dependency=afterok:${TRAIN_JOB} \
        --partition="$PARTITION" --cpus-per-task=2 --mem=4G --time=00:15:00 \
        --output="$RESULT_DIR/strat_trigger_%j.log" \
        --wrap="cd $SCRIPT_DIR && bash submit_stratified_100k.sh ${VARIANT} 0")
    echo "  Stage 3 eval trig  -> $EVAL_TRIGGER (after $TRAIN_JOB)"
    echo ""
done

echo "Encoder sweep submitted. Monitor: squeue -u \$USER --format='%.10i %.34j %.8T %.10M %R'"
