#!/bin/bash
# Phase B — BL-2 W=4000 + threshold features + (optional) GRPO post-step.
#
# Pipeline (fold-0 first; folds 1-4 gated on fold-0 results):
#   1. Precompute threshold features off sensor_embeddings_w4000/
#      -> sensor_embeddings_w4000_thresh/ (per-tick 16 extra features, 512+16=528)
#   2. Train AR-modern decoder over thresh embeddings (CE; the un-GRPO'd
#      reference) -> models/ar_modern_bl2_w4000_thresh_fold_0/ar_modern_best.pt
#   3a. GRPO post-step (state-stratified prompts, K=8 rollouts, NED reward,
#       KL to frozen reference) -> models/ar_modern_bl2_w4000_thresh_grpo_fold_0/grpo_best.pt
#   3b. GRPO post-step + per-state reward bonus (ablation)
#       -> models/ar_modern_bl2_w4000_thresh_grpo_sb_fold_0/grpo_best.pt
#   4. 100K stratified eval for each of the three checkpoints (thresh, grpo, grpo_sb)
#
# Reference: Phase A BL-2 W=4000 fold-0 NED-all = 0.0246; BL7+thresh
# 5-fold median = 0.0269 (current best). Kill criteria at fold-0:
# state-0 >= 0.080, state-2 >= 0.095, aggregate >= 0.046.

set -euo pipefail
SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

PARTITION="${PARTITION:-short}"
SEED="${SEED:-42}"
W="${W:-4000}"
SRC_EMB="$SCRIPT_DIR/sensor_embeddings_w${W}"
DST_EMB="$SCRIPT_DIR/sensor_embeddings_w${W}_thresh"

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")
FOLD_TRAIN_0=("2025-03-17_10-36-44" "2025-03-20_09-31-56")
ALL_SESS=("${FOLD_TRAIN_0[@]}" "${VAL_SESSIONS[@]}" "${TEST_SESSIONS[@]}")

THRESH_VARIANT="ar_modern_bl2_w${W}_thresh"
GRPO_VARIANT="ar_modern_bl2_w${W}_thresh_grpo"
GRPO_SB_VARIANT="ar_modern_bl2_w${W}_thresh_grpo_sb"

THRESH_DIR="$SCRIPT_DIR/models/${THRESH_VARIANT}_fold_0"
GRPO_DIR="$SCRIPT_DIR/models/${GRPO_VARIANT}_fold_0"
GRPO_SB_DIR="$SCRIPT_DIR/models/${GRPO_SB_VARIANT}_fold_0"
mkdir -p "$THRESH_DIR" "$GRPO_DIR" "$GRPO_SB_DIR" "$DST_EMB"

# ── Stage 1: thresh precompute (CPU; quick partition) ─────────────────────
PP_JOB=$(sbatch --parsable \
    --partition=quick --cpus-per-task=4 --mem=32G --time=01:00:00 \
    --job-name="bl2_w${W}_thresh_pp" \
    --output="${DST_EMB}/slurm_thresh_pp_%j.log" \
    --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
python3 -u precompute_thresh_features.py \
    --sessions ${ALL_SESS[*]} \
    --src-emb-dir $SRC_EMB \
    --dst-emb-dir $DST_EMB")
echo "Stage 1 thresh precompute -> $PP_JOB"

# ── Stage 2: AR-modern CE train over thresh emb (the reference for GRPO) ──
# d_sensor_emb = 512 (BL-2 W=4000) + 16 (thresh) = 528.
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$PP_JOB \
    --partition="$PARTITION" \
    --job-name="${THRESH_VARIANT}_f0_train" \
    --output="${THRESH_DIR}/slurm_train_%j.log" \
    run_ar_modern_train.sh \
    --seed "$SEED" \
    --train-sessions "${FOLD_TRAIN_0[@]}" \
    --val-sessions "${VAL_SESSIONS[@]}" \
    --emb-dir "$DST_EMB" \
    --d-sensor-emb 528 \
    --model-dir "$THRESH_DIR" \
    --max-steps 4000 --warmup-steps 500 --eval-every 500 --save-every 1000)
echo "Stage 2 AR-modern train  -> $TRAIN_JOB (after $PP_JOB)"

# ── Stage 3a: GRPO (stratified prompts, no state bonus) ───────────────────
POLICY="$THRESH_DIR/ar_modern_best.pt"
GRPO_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB \
    --partition="$PARTITION" \
    --job-name="${GRPO_VARIANT}_f0_grpo" \
    --output="${GRPO_DIR}/slurm_grpo_%j.log" \
    run_ar_modern_grpo.sh \
    --seed "$SEED" \
    --policy-ckpt "$POLICY" \
    --emb-dir "$DST_EMB" \
    --train-sessions "${FOLD_TRAIN_0[@]}" \
    --output-dir "$GRPO_DIR" \
    --d-sensor-emb 528 \
    --K 8 --batch-prompts 4 --num-steps 800 --lr 5e-6 --beta 0.04 --alpha 10.0)
echo "Stage 3a GRPO            -> $GRPO_JOB (after $TRAIN_JOB)"

# ── Stage 3b: GRPO with state bonus (ablation, parallel to 3a) ────────────
GRPO_SB_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB \
    --partition="$PARTITION" \
    --job-name="${GRPO_SB_VARIANT}_f0_grpo" \
    --output="${GRPO_SB_DIR}/slurm_grpo_%j.log" \
    run_ar_modern_grpo.sh \
    --seed "$SEED" \
    --policy-ckpt "$POLICY" \
    --emb-dir "$DST_EMB" \
    --train-sessions "${FOLD_TRAIN_0[@]}" \
    --output-dir "$GRPO_SB_DIR" \
    --d-sensor-emb 528 \
    --K 8 --batch-prompts 4 --num-steps 800 --lr 5e-6 --beta 0.04 --alpha 10.0 \
    --use-state-bonus --state-bonus 0.5)
echo "Stage 3b GRPO (state-bonus)-> $GRPO_SB_JOB (after $TRAIN_JOB)"

# ── Stage 4: 100K stratified eval for each ckpt ───────────────────────────
# AR-modern saves to model-dir/ar_modern_best.pt; GRPO saves to
# output-dir/grpo_best.pt. submit_stratified_100k.sh's case dispatch picks
# the right CKPT_NAME from the variant suffix (see the ar_modern_bl2_w*
# branch).
THRESH_EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB \
    --partition="$PARTITION" --cpus-per-task=2 --mem=4G --time=00:15:00 \
    --output="${THRESH_DIR}/strat_trigger_%j.log" \
    --wrap="cd $SCRIPT_DIR && bash submit_stratified_100k.sh $THRESH_VARIANT 0")
echo "Stage 4 thresh strat-eval-> $THRESH_EVAL (after $TRAIN_JOB)"

GRPO_EVAL=$(sbatch --parsable --dependency=afterok:$GRPO_JOB \
    --partition="$PARTITION" --cpus-per-task=2 --mem=4G --time=00:15:00 \
    --output="${GRPO_DIR}/strat_trigger_%j.log" \
    --wrap="cd $SCRIPT_DIR && bash submit_stratified_100k.sh $GRPO_VARIANT 0")
echo "Stage 4 grpo strat-eval  -> $GRPO_EVAL (after $GRPO_JOB)"

GRPO_SB_EVAL=$(sbatch --parsable --dependency=afterok:$GRPO_SB_JOB \
    --partition="$PARTITION" --cpus-per-task=2 --mem=4G --time=00:15:00 \
    --output="${GRPO_SB_DIR}/strat_trigger_%j.log" \
    --wrap="cd $SCRIPT_DIR && bash submit_stratified_100k.sh $GRPO_SB_VARIANT 0")
echo "Stage 4 grpo_sb strat-eval-> $GRPO_SB_EVAL (after $GRPO_SB_JOB)"

echo ""
echo "Phase B fold-0 submitted. Monitor:"
echo "  squeue -u \$USER --format='%.10i %.40j %.8T %.10M %R'"
