#!/bin/bash
# Re-evaluate existing AR-modern 5-fold checkpoints through the unified
# evaluate.py to pick up the per-FSM-state metric block. AR-modern is NOT
# retrained — uses the already-converged ar_modern_best.pt in
# models/ar_modern_fold_{i}/.

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")

echo "Submitting AR-modern 5-fold re-evaluation..."
JOBS=""

for fold_idx in 0 1 2 3 4; do
    FOLD_DIR="$SCRIPT_DIR/models/ar_modern_fold_${fold_idx}"
    RESULT_DIR="$SCRIPT_DIR/results/ar_modern_fold_${fold_idx}"
    mkdir -p "$RESULT_DIR"

    JOB=$(sbatch --parsable \
        --job-name="ar_modern_f${fold_idx}_eval" \
        --partition=short \
        --gres=gpu:H100:1 \
        --cpus-per-task=4 \
        --mem=64G \
        --time=06:00:00 \
        --exclude=gpu-6-[01-20] \
        --output="$RESULT_DIR/slurm_ar_modern_eval_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CKPT=$FOLD_DIR/ar_modern_best.pt
if [ ! -f \$CKPT ]; then
    CKPT=\$(ls -t $FOLD_DIR/ar_modern_final_*.pt 2>/dev/null | head -1)
fi
echo \"Using checkpoint: \$CKPT\"

echo '=== Fold $fold_idx (AR-modern): test ==='
python3 -u evaluate.py \
    --checkpoint \$CKPT \
    --split test \
    --sessions ${TEST_SESSIONS[*]} \
    --device cuda \
    --n-samples 30000 \
    --result-dir $RESULT_DIR

echo '=== Fold $fold_idx (AR-modern): val ==='
python3 -u evaluate.py \
    --checkpoint \$CKPT \
    --split val \
    --sessions ${VAL_SESSIONS[*]} \
    --device cuda \
    --n-samples 30000 \
    --result-dir $RESULT_DIR
")
    echo "  Fold $fold_idx: $JOB"
    JOBS="$JOBS $JOB"
done

echo ""
echo "All AR-modern eval jobs submitted:$JOBS"
