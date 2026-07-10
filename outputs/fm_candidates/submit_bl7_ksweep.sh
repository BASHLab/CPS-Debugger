#!/bin/bash
# E2 K-sweep submission: joint training of frozen BL-7 v1 backbone +
# trainable Perceiver pool + fresh AR-modern decoder, across K values.
#
# Usage:
#   bash submit_bl7_ksweep.sh                                    # all 6 variants × 5 folds
#   bash submit_bl7_ksweep.sh --K-list "4 8" --folds "0"          # smoke
#   K_VALUES="4 8 16 32 64" FOLDS="0 1 2 3 4" bash submit_bl7_ksweep.sh
#
# Variants:
#   K=4, 8, 16, 32, 64   → prefix decoder (sensor_emb is mean-pooled to (B,256))
#   K=axial              → cross-attn decoder with (B, 154, 512) prefix
#
# After all 5-fold training jobs complete, stratified-100K eval is auto-chained
# via submit_stratified_100k.sh per variant.

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

K_VALUES="${K_VALUES:-4 8 16 32 64 axial}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SEED="${SEED:-42}"
PARTITION="${PARTITION:-short}"
BL7_CKPT="$SCRIPT_DIR/models/bl7/bl7_best.pt"

# Sanity: encoder checkpoint exists
test -f "$BL7_CKPT" || { echo "ERROR: $BL7_CKPT missing" >&2; exit 1; }

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")
FOLD_TRAIN_0=("2025-03-17_10-36-44" "2025-03-20_09-31-56")
FOLD_TRAIN_1=("2025-03-17_10-51-58" "2025-03-20_09-46-30")
FOLD_TRAIN_2=("2025-03-17_11-06-39" "2025-03-20_10-03-00")
FOLD_TRAIN_3=("2025-03-18_12-39-10" "2025-03-19_11-10-13")
FOLD_TRAIN_4=("2025-03-19_10-05-47" "2025-03-19_10-37-56")

for K in $K_VALUES; do
    if [ "$K" == "axial" ]; then
        VARIANT_NAME="ar_modern_bl7_axial"
        EXTRA_ARGS="--bl7-encoder-ckpt $BL7_CKPT --bl7-axial --use-cross-attn --d-sensor-emb 512 --n-sensor-tokens 154 --cond-dropout-p 0.10"
    else
        # All K variants use cross-attn decoder so K tokens are preserved
        # (prefix decoder would mean-pool them, defeating the K-sweep test).
        VARIANT_NAME="ar_modern_bl7_k${K}"
        EXTRA_ARGS="--bl7-encoder-ckpt $BL7_CKPT --bl7-K $K --use-cross-attn --d-sensor-emb 256 --n-sensor-tokens $K --cond-dropout-p 0.10"
    fi

    TRAIN_JOBS=""
    declare -A TRAIN_BY_FOLD
    for fold in $FOLDS; do
        var="FOLD_TRAIN_${fold}[@]"
        TRAIN_SESS=("${!var}")

        FOLD_DIR="$SCRIPT_DIR/models/${VARIANT_NAME}_fold_${fold}"
        RESULT_DIR="$SCRIPT_DIR/results/${VARIANT_NAME}_fold_${fold}"
        mkdir -p "$FOLD_DIR" "$RESULT_DIR"

        TRAIN_JOB=$(sbatch --parsable \
            --partition="$PARTITION" \
            --gres=gpu:1 \
            --time=04:00:00 \
            --exclude=gpu-6-[01-20] \
            --job-name="${VARIANT_NAME}_f${fold}_train" \
            --output="$FOLD_DIR/slurm_${VARIANT_NAME}_train_%j.log" \
            run_ar_modern_train.sh \
            --seed "$SEED" \
            --train-sessions "${TRAIN_SESS[@]}" \
            --val-sessions "${VAL_SESSIONS[@]}" \
            --model-dir "$FOLD_DIR" \
            --max-steps 4000 --warmup-steps 500 --eval-every 500 --save-every 1000 \
            $EXTRA_ARGS)
        echo "  $VARIANT_NAME fold $fold train -> $TRAIN_JOB"
        TRAIN_JOBS="$TRAIN_JOBS $TRAIN_JOB"
        TRAIN_BY_FOLD[$fold]=$TRAIN_JOB
    done

    # Per-fold postprocess: extract encoder, then precompute embeddings into
    # a fold-specific dir. Each fold's joint-trained pool produces different
    # embeddings, so we MUST keep them separate (see submit_stratified_100k.sh
    # case for per-fold --emb-dir override).
    POSTPROC_JOBS=""
    for fold in $FOLDS; do
        FOLD_DIR="$SCRIPT_DIR/models/${VARIANT_NAME}_fold_${fold}"
        EMB_OUT_DIR="$SCRIPT_DIR/sensor_embeddings_${VARIANT_NAME}_fold_${fold}"
        mkdir -p "$EMB_OUT_DIR"
        TRAIN_DEP_JOB=${TRAIN_BY_FOLD[$fold]}
        PP=$(sbatch --parsable \
            --dependency=afterok:${TRAIN_DEP_JOB} \
            --partition="$PARTITION" \
            --gres=gpu:1 \
            --cpus-per-task=2 --mem=32G --time=01:00:00 \
            --exclude=gpu-6-[01-20] \
            --output="$FOLD_DIR/slurm_postprocess_%j.log" \
            --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
python3 -u extract_bl7_encoder_from_joint.py \
    --joint-ckpt $FOLD_DIR/ar_modern_best.pt \
    --output $FOLD_DIR/bl7_encoder.pt
python3 -u sensor_encoder_bl7.py \
    --ckpt $FOLD_DIR/bl7_encoder.pt \
    --sessions 2025-03-21_11-21-42 2025-03-25_13-23-42 2025-03-25_13-39-06 2025-03-26_11-03-04 \
    --output-dir $EMB_OUT_DIR \
    --batch-size 64 --device cuda
")
        POSTPROC_JOBS="$POSTPROC_JOBS $PP"
        echo "  $VARIANT_NAME fold $fold postprocess -> $PP"
    done

    # Chain stratified-100K eval after all postprocess jobs succeed.
    # Per-fold emb-dirs are handled inside submit_stratified_100k.sh's
    # case branch for ksweep variants (overrides EXTRA per fold).
    PP_DEP=$(echo $POSTPROC_JOBS | tr -s ' ' | sed 's/^ //; s/ /:/g')
    STRAT_TRIGGER=$(sbatch --parsable \
        --dependency=afterok:${PP_DEP} \
        --partition="$PARTITION" \
        --cpus-per-task=2 --mem=4G --time=00:15:00 \
        --output="$SCRIPT_DIR/results/strat_trigger_${VARIANT_NAME}_%j.log" \
        --wrap="cd $SCRIPT_DIR && CHUNK_TIME=12:00:00 bash submit_stratified_100k.sh ${VARIANT_NAME}")
    echo "  $VARIANT_NAME stratified-100K trigger -> $STRAT_TRIGGER"
    echo ""
done

echo "E2 K-sweep submitted. Monitor: squeue -u \$USER --format='%.10i %.30j %.8T %.10M %R'"
