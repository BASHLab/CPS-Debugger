#!/bin/bash
# Submit trivial-baseline eval for each of the 5 folds and 3 baselines
# (unigram, copy_modal, copy_previous). Uses the same val/test splits as the
# AR / AR-modern submit scripts, so baseline JSONs land next to model JSONs
# and the 5-fold collector can pair them up.
#
# Pinned to H100 via feature constraint so wall-clock noise matches the model
# eval runs. Baselines barely touch the GPU (MAUVE runs CPU k-means), but the
# matched hardware keeps the pipeline uniform.

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")

FOLD_TRAIN_0=("2025-03-17_10-36-44" "2025-03-20_09-31-56")
FOLD_TRAIN_1=("2025-03-17_10-51-58" "2025-03-20_09-46-30")
FOLD_TRAIN_2=("2025-03-17_11-06-39" "2025-03-20_10-03-00")
FOLD_TRAIN_3=("2025-03-18_12-39-10" "2025-03-19_11-10-13")
FOLD_TRAIN_4=("2025-03-19_10-05-47" "2025-03-19_10-37-56")

BASELINES=(unigram copy_modal copy_previous)

echo "Submitting trivial baselines (seed from evaluate.py DEFAULT_SEED)..."
echo "  Baselines: ${BASELINES[*]}"
JOBS=""

for fold_idx in 0 1 2 3 4; do
    var="FOLD_TRAIN_${fold_idx}[@]"
    TRAIN_SESS=("${!var}")

    RESULT_DIR="$SCRIPT_DIR/results/baselines_fold_${fold_idx}"
    mkdir -p "$RESULT_DIR"

    for baseline in "${BASELINES[@]}"; do
        JOB=$(sbatch --parsable \
            --job-name="base_${baseline}_f${fold_idx}" \
            --partition=short \
            --gres=gpu:H100:1 \
            --cpus-per-task=4 \
            --mem=32G \
            --time=04:00:00 \
            --exclude=gpu-6-[01-20] \
            --output="$RESULT_DIR/slurm_${baseline}_%j.log" \
            --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo '=== Fold $fold_idx ($baseline): test ==='
python3 -u evaluate.py \
    --baseline $baseline \
    --split test \
    --sessions ${TEST_SESSIONS[*]} \
    --baseline-train-sessions ${TRAIN_SESS[*]} \
    --device cuda \
    --n-samples 30000 \
    --result-dir $RESULT_DIR

echo '=== Fold $fold_idx ($baseline): val ==='
python3 -u evaluate.py \
    --baseline $baseline \
    --split val \
    --sessions ${VAL_SESSIONS[*]} \
    --baseline-train-sessions ${TRAIN_SESS[*]} \
    --device cuda \
    --n-samples 30000 \
    --result-dir $RESULT_DIR
")
        echo "  Fold $fold_idx / $baseline: $JOB"
        JOBS="$JOBS $JOB"
    done
done

echo ""
echo "All baseline jobs submitted:$JOBS"
