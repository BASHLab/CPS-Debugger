#!/bin/bash
# Phase A — BL-2 (Chronos-2 + AR-modern) window sweep.
#
# For each W in {1000, 1500, 4000}, run an end-to-end chain (afterok-gated):
#   1. Finetune Chronos-2 at window_size=W  (slimllm_bw env: needs transformers
#      >= 4.36 for chronos-forecasting 2.2.2 'save_only_model' kwarg).
#      Strategy B: batch * grad_accum = 128 (effective batch == legacy);
#      LR=1e-4, num_steps=5000 — identical optimizer dynamics, only window varies.
#   2. Precompute per-tick embeddings  (slimllm_bw env, --window-size W).
#   3. Train AR-modern decoder fold-0  (slimllm env via run_ar_modern_train.sh).
#   4. 100K stratified eval fold-0     (slimllm env via submit_stratified_100k.sh).
#
# Reference: 500-tick BL-2 5-fold median NED-all = 0.0291 (existing leaderboard).
# Per-state kill criteria at fold-0: s0>=0.080, s2>=0.095, agg>=0.046 ⇒ abandon.

set -euo pipefail
SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

PARTITION="${PARTITION:-short}"
SEED="${SEED:-42}"

# (W, per-step batch, grad-accum) — effective batch = 128 for all.
SWEEP="${SWEEP:-1000:64:2 1500:32:4 4000:16:8}"

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")
FOLD_TRAIN_0=("2025-03-17_10-36-44" "2025-03-20_09-31-56")

for entry in $SWEEP; do
    IFS=':' read -r W BATCH ACCUM <<< "$entry"
    VARIANT="ar_modern_bl2_w${W}"
    CHRONOS_CKPT_DIR="$SCRIPT_DIR/models/chronos2_finetuned_w${W}"
    EMB_DIR="$SCRIPT_DIR/sensor_embeddings_w${W}"
    FOLD_DIR="$SCRIPT_DIR/models/${VARIANT}_fold_0"
    RD="$SCRIPT_DIR/results/${VARIANT}_fold_0"
    mkdir -p "$CHRONOS_CKPT_DIR" "$EMB_DIR" "$FOLD_DIR" "$RD"

    echo "── W=$W (batch=$BATCH × accum=$ACCUM = $((BATCH*ACCUM)) effective) ──"

    # ── Stage 1: finetune Chronos-2 (slimllm_bw) ───────────────────────────
    FT_JOB=$(sbatch --parsable \
        --partition="$PARTITION" --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=12:00:00 \
        --exclude=gpu-6-[01-20] \
        --job-name="bl2_w${W}_finetune" \
        --output="${CHRONOS_CKPT_DIR}/slurm_finetune_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm_bw/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -u finetune_chronos.py --device cuda --window-size $W \
    --batch-size $BATCH --grad-accum-steps $ACCUM --num-steps 5000 \
    --output-dir $CHRONOS_CKPT_DIR")
    echo "  Stage 1 finetune  -> $FT_JOB"

    # ── Stage 2: precompute embeddings (slimllm_bw, afterok) ───────────────
    # Smoke established W=1000 batch=32 -> ~331 ticks/s. W=4000 with O(seq^2)
    # attention is ~10x slower; bumped batch to 64 (still fits H100/80GB) and
    # time to 16h for the W=4000 worst case. H100 pinned for consistent memory.
    ALL_SESS=("${FOLD_TRAIN_0[@]}" "${VAL_SESSIONS[@]}" "${TEST_SESSIONS[@]}")
    PP_JOB=$(sbatch --parsable --dependency=afterok:$FT_JOB \
        --partition="$PARTITION" --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=16:00:00 \
        --exclude=gpu-6-[01-20] \
        --job-name="bl2_w${W}_precompute" \
        --output="${EMB_DIR}/slurm_precompute_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm_bw/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -u sensor_encoder.py --sessions ${ALL_SESS[*]} \
    --window-size $W --chronos-ckpt $CHRONOS_CKPT_DIR --output-dir $EMB_DIR \
    --batch-size 64 --device cuda")
    echo "  Stage 2 precompute-> $PP_JOB (after $FT_JOB)"

    # ── Stage 3: AR-modern decoder train fold-0 (slimllm, afterok) ─────────
    TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$PP_JOB \
        --partition="$PARTITION" \
        --job-name="${VARIANT}_f0_train" \
        --output="${FOLD_DIR}/slurm_${VARIANT}_train_%j.log" \
        run_ar_modern_train.sh \
        --seed "$SEED" \
        --train-sessions "${FOLD_TRAIN_0[@]}" \
        --val-sessions "${VAL_SESSIONS[@]}" \
        --emb-dir "$EMB_DIR" \
        --d-sensor-emb 512 \
        --model-dir "$FOLD_DIR" \
        --max-steps 4000 --warmup-steps 500 --eval-every 500 --save-every 1000)
    echo "  Stage 3 AR train  -> $TRAIN_JOB (after $PP_JOB)"

    # ── Stage 4: 100K stratified eval fold-0 (afterok) ─────────────────────
    EVAL_TRIGGER=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB \
        --partition="$PARTITION" --cpus-per-task=2 --mem=4G --time=00:15:00 \
        --output="${RD}/strat_trigger_%j.log" \
        --wrap="cd $SCRIPT_DIR && bash submit_stratified_100k.sh $VARIANT 0")
    echo "  Stage 4 eval trig -> $EVAL_TRIGGER (after $TRAIN_JOB)"
    echo ""
done

echo "Phase A window sweep submitted. Monitor: squeue -u \$USER --format='%.10i %.34j %.8T %.10M %R'"
