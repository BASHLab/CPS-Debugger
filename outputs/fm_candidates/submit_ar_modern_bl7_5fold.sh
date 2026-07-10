#!/bin/bash
# Submit 5-fold AR-modern training + evaluation on BL-7 sensor embeddings.
# Mirrors submit_ar_modern_5fold.sh exactly except:
#   --emb-dir       points at sensor_embeddings_bl7/  (BL-7 dense embeddings,
#                                                      mean-pooled to (N, 256)
#                                                      by data.py at load time)
#   --d-sensor-emb  256  (matches BL-7's pooled output dim)
#   model-dir       ar_modern_bl7_fold_<i>  (separate checkpoint namespace)
#   result-dir      ar_modern_bl7_fold_<i>
#
# Prerequisite: BL-7 pretrain has finished AND sensor_encoder_bl7.py has been
# run on all 13 sessions (TRAIN ∪ VAL ∪ TEST), populating
# sensor_embeddings_bl7/<sess>_emb.npy with shape (N_ticks, 4, 256).

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

EMB_DIR="${EMB_DIR:-$SCRIPT_DIR/sensor_embeddings_bl7}"
D_SENSOR_EMB="${D_SENSOR_EMB:-256}"
# Variant naming — override for the 2x2 factorial cells.
#   Cell B (Tier 2):       VARIANT_NAME=ar_modern_bl7_v2
#   Cell A (Tier 1A.1):    VARIANT_NAME=ar_modern_bl7_cross
#   Cell C (both):         VARIANT_NAME=ar_modern_bl7_v2_cross
VARIANT_NAME="${VARIANT_NAME:-ar_modern_bl7}"
# Extra args to ar_modern_train.py (e.g. --use-cross-attn for cells A and C)
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

# Fixed val/test across all folds
VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")

# Same 5 train folds as MDLM and AR-modern (Chronos-2)
FOLD_TRAIN_0=("2025-03-17_10-36-44" "2025-03-20_09-31-56")
FOLD_TRAIN_1=("2025-03-17_10-51-58" "2025-03-20_09-46-30")
FOLD_TRAIN_2=("2025-03-17_11-06-39" "2025-03-20_10-03-00")
FOLD_TRAIN_3=("2025-03-18_12-39-10" "2025-03-19_11-10-13")
FOLD_TRAIN_4=("2025-03-19_10-05-47" "2025-03-19_10-37-56")

SEED="${SEED:-42}"
PARTITION="${PARTITION:-short}"

# Verify the BL-7 embeddings exist before submitting
ALL_SESS=("${VAL_SESSIONS[@]}" "${TEST_SESSIONS[@]}")
for fold_idx in 0 1 2 3 4; do
    var="FOLD_TRAIN_${fold_idx}[@]"
    ALL_SESS+=("${!var}")
done
MISSING=()
for sess in "${ALL_SESS[@]}"; do
    if [ ! -f "$EMB_DIR/${sess}_emb.npy" ]; then
        MISSING+=("$sess")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: BL-7 embeddings missing for these sessions in $EMB_DIR:" >&2
    for s in "${MISSING[@]}"; do echo "  - $s" >&2; done
    echo "Run sensor_encoder_bl7.py first. Aborting." >&2
    exit 1
fi
echo "All ${#ALL_SESS[@]} sessions have BL-7 embeddings under $EMB_DIR."
echo "Using d_sensor_emb=$D_SENSOR_EMB, partition=$PARTITION, seed=$SEED"

echo ""
echo "Submitting 5-fold AR-modern (BL-7 embeddings) training (seed=$SEED)..."
echo "  Val  (fixed): ${VAL_SESSIONS[*]}"
echo "  Test (fixed): ${TEST_SESSIONS[*]}"
TRAIN_JOBS=""

FOLDS="${FOLDS:-0 1 2 3 4}"
for fold_idx in $FOLDS; do
    var="FOLD_TRAIN_${fold_idx}[@]"
    TRAIN_SESS=("${!var}")

    FOLD_DIR="$SCRIPT_DIR/models/${VARIANT_NAME}_fold_${fold_idx}"
    RESULT_DIR="$SCRIPT_DIR/results/${VARIANT_NAME}_fold_${fold_idx}"
    mkdir -p "$FOLD_DIR" "$RESULT_DIR"

    echo ""
    echo "Fold $fold_idx: train=${TRAIN_SESS[*]}"

    TRAIN_JOB=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name="${VARIANT_NAME}_f${fold_idx}_train" \
        --output="$FOLD_DIR/slurm_${VARIANT_NAME}_train_%j.log" \
        run_ar_modern_train.sh \
        --seed "$SEED" \
        --train-sessions "${TRAIN_SESS[@]}" \
        --val-sessions "${VAL_SESSIONS[@]}" \
        --emb-dir "$EMB_DIR" \
        --d-sensor-emb "$D_SENSOR_EMB" \
        --model-dir "$FOLD_DIR" \
        --max-steps 4000 --warmup-steps 500 --eval-every 500 --save-every 1000 \
        $EXTRA_TRAIN_ARGS)
    echo "  Train job: $TRAIN_JOB"

    EVAL_JOB=$(sbatch --parsable \
        --dependency=afterok:${TRAIN_JOB} \
        --job-name="${VARIANT_NAME}_f${fold_idx}_eval" \
        --partition="$PARTITION" \
        --gres=gpu:H100:1 \
        --cpus-per-task=4 \
        --mem=64G \
        --time=04:00:00 \
        --exclude=gpu-6-[01-20] \
        --output="$RESULT_DIR/slurm_ar_modern_bl7_eval_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CKPT=$FOLD_DIR/ar_modern_best.pt
if [ ! -f \$CKPT ]; then
    CKPT=\$(ls -t $FOLD_DIR/ar_modern_final_*.pt 2>/dev/null | head -1)
fi

echo '=== Fold $fold_idx (AR-modern + BL-7): Evaluating test split ==='
python3 -u evaluate.py \
    --checkpoint \$CKPT \
    --split test \
    --sessions ${TEST_SESSIONS[*]} \
    --device cuda \
    --emb-dir $EMB_DIR \
    --n-samples 1000 \
    --result-dir $RESULT_DIR

echo '=== Fold $fold_idx (AR-modern + BL-7): Evaluating val split ==='
python3 -u evaluate.py \
    --checkpoint \$CKPT \
    --split val \
    --sessions ${VAL_SESSIONS[*]} \
    --device cuda \
    --emb-dir $EMB_DIR \
    --n-samples 1000 \
    --result-dir $RESULT_DIR
")
    echo "  Eval job:  $EVAL_JOB (after $TRAIN_JOB)"

    TRAIN_JOBS="$TRAIN_JOBS $TRAIN_JOB"
done

echo ""
echo "All AR-modern (BL-7) folds submitted. Train jobs:$TRAIN_JOBS"

# Chain stratified-100K eval after all 5 train jobs succeed.
# (Fast 1000-sample eval already runs above for cluster-monitoring sake;
# this is the headline number.)
DEP=$(echo $TRAIN_JOBS | tr -s ' ' | sed 's/^ //; s/ /:/g')
STRAT_TRIGGER=$(sbatch --parsable \
    --dependency=afterok:${DEP} \
    --partition="$PARTITION" \
    --cpus-per-task=2 --mem=4G --time=00:15:00 \
    --output="$SCRIPT_DIR/results/strat_trigger_${VARIANT_NAME}_%j.log" \
    --wrap="cd $SCRIPT_DIR && bash submit_stratified_100k.sh ${VARIANT_NAME}")
echo "Stratified-100K trigger: $STRAT_TRIGGER (afterok:$DEP)"
echo "Monitor: squeue -u \$USER --format='%.10i %.25j %.8T %.10M %.6D %R'"
