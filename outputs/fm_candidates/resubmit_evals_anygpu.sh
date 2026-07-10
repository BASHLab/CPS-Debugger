#!/bin/bash
# Resubmit AR-variant evals on ANY non-Blackwell GPU (BL-3/4/6 + BL-2 AR-modern).
# H100 SXM5 is heavily contended; only 8 of those exist cluster-wide. AR sampling
# + metric pipeline is light enough that A100/L40S/A30 finish in well under 6h.
# Note: ms/tick measurements from this batch are mixed-hardware. For an apples-
# to-apples timing table, re-run a single fold on H100 later.

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")

submit_eval() {
    local variant=$1
    local fold_idx=$2
    local fold_dir="$SCRIPT_DIR/models/${variant}_fold_${fold_idx}"
    local result_dir="$SCRIPT_DIR/results/${variant}_fold_${fold_idx}"
    mkdir -p "$result_dir"
    sbatch --parsable \
        --job-name="${variant}_f${fold_idx}_eval" \
        --partition=short \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=64G \
        --time=06:00:00 \
        --exclude=gpu-6-[01-20] \
        --output="$result_dir/slurm_${variant}_eval_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

CKPT=$fold_dir/${variant}_best.pt
if [ ! -f \$CKPT ]; then
    CKPT=\$(ls -t $fold_dir/${variant}_final_*.pt 2>/dev/null | head -1)
fi
echo \"Using checkpoint: \$CKPT\"

echo '=== Fold $fold_idx ($variant): test ==='
python3 -u evaluate.py --checkpoint \$CKPT --split test --sessions ${TEST_SESSIONS[*]} --device cuda --n-samples 30000 --num-workers 0 --result-dir $result_dir

echo '=== Fold $fold_idx ($variant): val ==='
python3 -u evaluate.py --checkpoint \$CKPT --split val --sessions ${VAL_SESSIONS[*]} --device cuda --n-samples 30000 --num-workers 0 --result-dir $result_dir
"
}

# AR-modern variant uses different checkpoint name
submit_ar_modern_eval() {
    local fold_idx=$1
    local fold_dir="$SCRIPT_DIR/models/ar_modern_fold_${fold_idx}"
    local result_dir="$SCRIPT_DIR/results/ar_modern_fold_${fold_idx}"
    mkdir -p "$result_dir"
    sbatch --parsable \
        --job-name="ar_modern_f${fold_idx}_eval" \
        --partition=short \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=64G \
        --time=06:00:00 \
        --exclude=gpu-6-[01-20] \
        --output="$result_dir/slurm_ar_modern_eval_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

CKPT=$fold_dir/ar_modern_best.pt
if [ ! -f \$CKPT ]; then
    CKPT=\$(ls -t $fold_dir/ar_modern_final_*.pt 2>/dev/null | head -1)
fi
echo \"Using checkpoint: \$CKPT\"

echo '=== Fold $fold_idx (AR-modern): test ==='
python3 -u evaluate.py --checkpoint \$CKPT --split test --sessions ${TEST_SESSIONS[*]} --device cuda --n-samples 30000 --num-workers 0 --result-dir $result_dir

echo '=== Fold $fold_idx (AR-modern): val ==='
python3 -u evaluate.py --checkpoint \$CKPT --split val --sessions ${VAL_SESSIONS[*]} --device cuda --n-samples 30000 --num-workers 0 --result-dir $result_dir
"
}

JOBS=""
for variant in bl3 bl4 bl6; do
    for fold_idx in 0 1 2 3 4; do
        JOB=$(submit_eval "$variant" "$fold_idx")
        echo "  $variant fold $fold_idx -> $JOB"
        JOBS="$JOBS $JOB"
    done
done
for fold_idx in 0 1 2 3 4; do
    JOB=$(submit_ar_modern_eval "$fold_idx")
    echo "  ar_modern fold $fold_idx -> $JOB"
    JOBS="$JOBS $JOB"
done
echo ""
echo "All eval jobs:$JOBS"
