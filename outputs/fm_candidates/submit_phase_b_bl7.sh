#!/bin/bash
# Phase B sibling — GRPO on BL-7 (CPS-JEPA, 500-tick) + threshold features.
#
# Targets the CURRENT OVERALL BEST baseline (BL7+thresh, 5-fold median NED-all
# 0.0269; state-2 NED 0.0738 is the weakest spot, exactly what stratified GRPO
# should attack). The BL-2 chain in submit_phase_b.sh targets BL-2 W=4000 in
# parallel — same GRPO loop, different encoder.
#
# AR-modern CE checkpoint AND sensor_embeddings_bl7_thresh/ already exist, so
# this chain skips stages 1-2 and goes directly to GRPO + eval.
#
# Pipeline (fold-0 first; folds 1-4 gated):
#   1. GRPO (stratified prompts, no state bonus)
#       -> models/ar_modern_bl7_e1thresh_grpo_fold_0/grpo_best.pt
#   2. GRPO + per-state reward bonus (ablation)
#       -> models/ar_modern_bl7_e1thresh_grpo_sb_fold_0/grpo_best.pt
#   3. 100K stratified eval for each.

set -euo pipefail
SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

PARTITION="${PARTITION:-short}"
SEED="${SEED:-42}"
EMB_DIR="$SCRIPT_DIR/sensor_embeddings_bl7_thresh"
POLICY="$SCRIPT_DIR/models/ar_modern_bl7_e1thresh_fold_0/ar_modern_best.pt"

if [[ ! -f "$POLICY" ]]; then
    echo "ERROR: missing AR-modern CE ckpt: $POLICY" >&2
    exit 2
fi

FOLD_TRAIN_0=("2025-03-17_10-36-44" "2025-03-20_09-31-56")
GRPO_VARIANT="ar_modern_bl7_e1thresh_grpo"
GRPO_SB_VARIANT="ar_modern_bl7_e1thresh_grpo_sb"
GRPO_DIR="$SCRIPT_DIR/models/${GRPO_VARIANT}_fold_0"
GRPO_SB_DIR="$SCRIPT_DIR/models/${GRPO_SB_VARIANT}_fold_0"
mkdir -p "$GRPO_DIR" "$GRPO_SB_DIR"

# d_sensor_emb = 256 (BL-7 K=4 tokens) + 16 (thresh) = 272.
# ── Stage 1: GRPO (stratified prompts only) ─────────────────────────────
GRPO_JOB=$(sbatch --parsable \
    --partition="$PARTITION" \
    --job-name="${GRPO_VARIANT}_f0_grpo" \
    --output="${GRPO_DIR}/slurm_grpo_%j.log" \
    run_ar_modern_grpo.sh \
    --seed "$SEED" \
    --policy-ckpt "$POLICY" \
    --emb-dir "$EMB_DIR" \
    --train-sessions "${FOLD_TRAIN_0[@]}" \
    --output-dir "$GRPO_DIR" \
    --d-sensor-emb 272 \
    --K 8 --batch-prompts 4 --num-steps 800 --lr 5e-6 --beta 0.04 --alpha 10.0)
echo "Stage 1 GRPO              -> $GRPO_JOB"

# ── Stage 2: GRPO with state bonus (parallel) ───────────────────────────
GRPO_SB_JOB=$(sbatch --parsable \
    --partition="$PARTITION" \
    --job-name="${GRPO_SB_VARIANT}_f0_grpo" \
    --output="${GRPO_SB_DIR}/slurm_grpo_%j.log" \
    run_ar_modern_grpo.sh \
    --seed "$SEED" \
    --policy-ckpt "$POLICY" \
    --emb-dir "$EMB_DIR" \
    --train-sessions "${FOLD_TRAIN_0[@]}" \
    --output-dir "$GRPO_SB_DIR" \
    --d-sensor-emb 272 \
    --K 8 --batch-prompts 4 --num-steps 800 --lr 5e-6 --beta 0.04 --alpha 10.0 \
    --use-state-bonus --state-bonus 0.5)
echo "Stage 2 GRPO (state-bonus)-> $GRPO_SB_JOB"

# ── Stage 3: 100K stratified eval for each ckpt ─────────────────────────
GRPO_EVAL=$(sbatch --parsable --dependency=afterok:$GRPO_JOB \
    --partition="$PARTITION" --cpus-per-task=2 --mem=4G --time=00:15:00 \
    --output="${GRPO_DIR}/strat_trigger_%j.log" \
    --wrap="cd $SCRIPT_DIR && bash submit_stratified_100k.sh $GRPO_VARIANT 0")
echo "Stage 3 grpo strat-eval   -> $GRPO_EVAL (after $GRPO_JOB)"

GRPO_SB_EVAL=$(sbatch --parsable --dependency=afterok:$GRPO_SB_JOB \
    --partition="$PARTITION" --cpus-per-task=2 --mem=4G --time=00:15:00 \
    --output="${GRPO_SB_DIR}/strat_trigger_%j.log" \
    --wrap="cd $SCRIPT_DIR && bash submit_stratified_100k.sh $GRPO_SB_VARIANT 0")
echo "Stage 3 grpo_sb strat-eval-> $GRPO_SB_EVAL (after $GRPO_SB_JOB)"

echo ""
echo "Phase B (BL-7 sibling) fold-0 submitted. Monitor:"
echo "  squeue -u \$USER --format='%.10i %.40j %.8T %.10M %R'"
