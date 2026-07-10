#!/bin/bash
# Submit 5-fold AR baseline training + evaluation.
# Split rationale identical to submit_mdlm_5fold.sh (same val/test).
# All 5 folds run in parallel.

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

# Fixed val/test across all folds
VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")

# Same 5 train folds as MDLM
FOLD_TRAIN_0=("2025-03-17_10-36-44" "2025-03-20_09-31-56")
FOLD_TRAIN_1=("2025-03-17_10-51-58" "2025-03-20_09-46-30")
FOLD_TRAIN_2=("2025-03-17_11-06-39" "2025-03-20_10-03-00")
FOLD_TRAIN_3=("2025-03-18_12-39-10" "2025-03-19_11-10-13")
FOLD_TRAIN_4=("2025-03-19_10-05-47" "2025-03-19_10-37-56")
# Unused: 2025-03-19_10-20-35

SEED="${SEED:-42}"

echo "Submitting 5-fold AR training (seed=$SEED)..."
echo "  Val  (fixed): ${VAL_SESSIONS[*]}"
echo "  Test (fixed): ${TEST_SESSIONS[*]}"
TRAIN_JOBS=""

for fold_idx in 0 1 2 3 4; do
    var="FOLD_TRAIN_${fold_idx}[@]"
    TRAIN_SESS=("${!var}")

    FOLD_DIR="$SCRIPT_DIR/models/ar_fold_${fold_idx}"
    RESULT_DIR="$SCRIPT_DIR/results/ar_fold_${fold_idx}"
    mkdir -p "$FOLD_DIR" "$RESULT_DIR"

    echo ""
    echo "Fold $fold_idx: train=${TRAIN_SESS[*]}"

    # AR converges by ~3K steps; 4000 gives margin past best, 500 warmup.
    TRAIN_JOB=$(sbatch --parsable \
        --job-name="ar_f${fold_idx}_train" \
        --output="$FOLD_DIR/slurm_ar_train_%j.log" \
        run_ar_train.sh \
        --seed "$SEED" \
        --train-sessions "${TRAIN_SESS[@]}" \
        --val-sessions "${VAL_SESSIONS[@]}" \
        --model-dir "$FOLD_DIR" \
        --max-steps 4000 --warmup-steps 500 --eval-every 500 --save-every 1000)
    echo "  Train job: $TRAIN_JOB"

    # Eval pinned to H100 (via feature constraint) for matched timing.
    EVAL_JOB=$(sbatch --parsable \
        --dependency=afterok:${TRAIN_JOB} \
        --job-name="ar_f${fold_idx}_eval" \
        --partition=short \
        --gres=gpu:H100:1 \
        --cpus-per-task=4 \
        --mem=64G \
        --time=04:00:00 \
        --exclude=gpu-6-[01-20] \
        --output="$RESULT_DIR/slurm_ar_eval_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CKPT=$FOLD_DIR/ar_best.pt
if [ ! -f \$CKPT ]; then
    CKPT=\$(ls -t $FOLD_DIR/ar_final_*.pt 2>/dev/null | head -1)
fi

echo '=== Fold $fold_idx (AR): Evaluating test split (held-out, Chronos FT never saw) ==='
python3 -u evaluate.py \
    --checkpoint \$CKPT \
    --split test \
    --sessions ${TEST_SESSIONS[*]} \
    --device cuda \
    --n-samples 1000 \
    --result-dir $RESULT_DIR

echo '=== Fold $fold_idx (AR): Evaluating val split ==='
python3 -u evaluate.py \
    --checkpoint \$CKPT \
    --split val \
    --sessions ${VAL_SESSIONS[*]} \
    --device cuda \
    --n-samples 1000 \
    --result-dir $RESULT_DIR
")
    echo "  Eval job:  $EVAL_JOB (after $TRAIN_JOB)"

    TRAIN_JOBS="$TRAIN_JOBS $TRAIN_JOB"
done

echo ""
echo "All AR folds submitted. Train jobs:$TRAIN_JOBS"
echo "Monitor: squeue -u \$USER --format='%.10i %.25j %.8T %.10M %.6D %R'"
