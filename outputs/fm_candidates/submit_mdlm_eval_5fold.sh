#!/bin/bash
# Re-evaluate existing MDLM 5-fold checkpoints. Splits test and val into
# separate jobs (each fits in the 6h SLURM cap; back-to-back runs of both
# splits in one job hit the timeout). MDLM sampling steps reduced to 200
# (5x speedup; 200 is standard for MDLM eval — full 1000 is overkill on this
# mode-collapsed corpus and was the actual cause of the 6h timeouts).

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")
SAMPLING_STEPS=200

echo "Submitting MDLM 5-fold re-evaluation (split test/val, $SAMPLING_STEPS sampling steps)..."
JOBS=""

for fold_idx in 0 1 2 3 4; do
    FOLD_DIR="$SCRIPT_DIR/models/fold_${fold_idx}"
    RESULT_DIR="$SCRIPT_DIR/results/mdlm_fold_${fold_idx}"
    mkdir -p "$RESULT_DIR"

    for split in test val; do
        if [ "$split" = "test" ]; then
            SESSIONS_LIST="${TEST_SESSIONS[*]}"
        else
            SESSIONS_LIST="${VAL_SESSIONS[*]}"
        fi
        JOB=$(sbatch --parsable \
            --job-name="mdlm_f${fold_idx}_${split}" \
            --partition=short \
            --gres=gpu:H100:1 \
            --cpus-per-task=4 \
            --mem=64G \
            --time=06:00:00 \
            --exclude=gpu-6-[01-20] \
            --output="$RESULT_DIR/slurm_mdlm_${split}_%j.log" \
            --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CKPT=$FOLD_DIR/bl2_best.pt
if [ ! -f \$CKPT ]; then
    CKPT=\$(ls -t $FOLD_DIR/bl2_final_*.pt 2>/dev/null | head -1)
fi
echo \"Using checkpoint: \$CKPT\"

echo '=== Fold $fold_idx (MDLM): $split ==='
python3 -u evaluate.py \
    --checkpoint \$CKPT \
    --split $split \
    --sessions $SESSIONS_LIST \
    --device cuda \
    --n-samples 30000 \
    --sampling-steps $SAMPLING_STEPS \
    --result-dir $RESULT_DIR
")
        echo "  Fold $fold_idx / $split: $JOB"
        JOBS="$JOBS $JOB"
    done
done

echo ""
echo "All MDLM eval jobs submitted:$JOBS"
