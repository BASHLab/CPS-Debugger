#!/bin/bash
# Submit 5-fold training + evaluation for one of {bl3, bl4, bl5}.
# Same val/test/train fold layout as submit_ar_modern_5fold.sh so the
# resulting JSONs slot into the same collector.
#
# Usage:
#   bash submit_bl_video_5fold.sh bl3
#   bash submit_bl_video_5fold.sh bl4
#   bash submit_bl_video_5fold.sh bl5

VARIANT="${1:-bl3}"
case "$VARIANT" in
  bl3|bl4|bl5|bl6|bl6b) ;;
  *) echo "ERROR: variant must be one of {bl3, bl4, bl5, bl6, bl6b}, got: $VARIANT" >&2; exit 1 ;;
esac

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")

FOLD_TRAIN_0=("2025-03-17_10-36-44" "2025-03-20_09-31-56")
FOLD_TRAIN_1=("2025-03-17_10-51-58" "2025-03-20_09-46-30")
FOLD_TRAIN_2=("2025-03-17_11-06-39" "2025-03-20_10-03-00")
FOLD_TRAIN_3=("2025-03-18_12-39-10" "2025-03-19_11-10-13")
FOLD_TRAIN_4=("2025-03-19_10-05-47" "2025-03-19_10-37-56")

SEED="${SEED:-42}"
PARTITION="${PARTITION:-short}"
DEPENDENCY="${DEPENDENCY:-}"   # optional, e.g. afterok:JOB1:JOB2:...
echo "Submitting 5-fold $VARIANT training (seed=$SEED, partition=$PARTITION, dep=${DEPENDENCY:-none})..."

TRAIN_JOBS=""
for fold_idx in 0 1 2 3 4; do
    var="FOLD_TRAIN_${fold_idx}[@]"
    TRAIN_SESS=("${!var}")

    FOLD_DIR="$SCRIPT_DIR/models/${VARIANT}_fold_${fold_idx}"
    RESULT_DIR="$SCRIPT_DIR/results/${VARIANT}_fold_${fold_idx}"
    mkdir -p "$FOLD_DIR" "$RESULT_DIR"

    echo ""
    echo "Fold $fold_idx: train=${TRAIN_SESS[*]}"

    DEP_ARG=""
    [ -n "$DEPENDENCY" ] && DEP_ARG="--dependency=$DEPENDENCY"
    TRAIN_JOB=$(sbatch --parsable \
        $DEP_ARG \
        --partition="$PARTITION" \
        --job-name="${VARIANT}_f${fold_idx}_train" \
        --output="$FOLD_DIR/slurm_${VARIANT}_train_%j.log" \
        run_bl_video_train.sh \
        --variant "$VARIANT" \
        --seed "$SEED" \
        --train-sessions "${TRAIN_SESS[@]}" \
        --val-sessions "${VAL_SESSIONS[@]}" \
        --model-dir "$FOLD_DIR" \
        --max-steps 4000 --warmup-steps 500 --eval-every 500 --save-every 1000)
    echo "  Train job: $TRAIN_JOB"

    EVAL_JOB=$(sbatch --parsable \
        --dependency=afterok:${TRAIN_JOB} \
        --job-name="${VARIANT}_f${fold_idx}_eval" \
        --partition=short \
        --gres=gpu:H100:1 \
        --cpus-per-task=4 \
        --mem=64G \
        --time=06:00:00 \
        --exclude=gpu-6-[01-20] \
        --output="$RESULT_DIR/slurm_${VARIANT}_eval_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CKPT=$FOLD_DIR/${VARIANT}_best.pt
if [ ! -f \$CKPT ]; then
    CKPT=\$(ls -t $FOLD_DIR/${VARIANT}_final_*.pt 2>/dev/null | head -1)
fi
echo \"Using checkpoint: \$CKPT\"

echo '=== Fold $fold_idx ($VARIANT): test ==='
python3 -u evaluate.py --checkpoint \$CKPT --split test --sessions ${TEST_SESSIONS[*]} --device cuda --n-samples 30000 --result-dir $RESULT_DIR

echo '=== Fold $fold_idx ($VARIANT): val ==='
python3 -u evaluate.py --checkpoint \$CKPT --split val --sessions ${VAL_SESSIONS[*]} --device cuda --n-samples 30000 --result-dir $RESULT_DIR
")
    echo "  Eval job:  $EVAL_JOB (after $TRAIN_JOB)"

    TRAIN_JOBS="$TRAIN_JOBS $TRAIN_JOB"
done

echo ""
echo "All $VARIANT folds submitted. Train jobs:$TRAIN_JOBS"
