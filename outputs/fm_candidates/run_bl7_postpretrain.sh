#!/bin/bash
# Auto-pipeline: runs after BL-7 pretrain completes.
#   1. Precompute BL-7 dense embeddings for all 13 sessions
#   2. Submit the 5-fold AR-modern + BL-7 training + eval jobs
#
# Submit with:
#   sbatch --gres=gpu:H100:1 --time=08:00:00 \
#          --dependency=afterok:<bl7_pretrain_jobid> \
#          --output=results/bl7_postpretrain_%j.log \
#          run_bl7_postpretrain.sh
#
# Note: precompute uses the same env as pretrain. If BL-7 pretrain ran on
# Blackwell with --use-blackwell-env, pass it here too. For H100/H200 the
# default slimllm env is correct.

#SBATCH --job-name=bl7_postpretrain
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

USE_BLACKWELL=0
for arg in "$@"; do
    if [ "$arg" == "--use-blackwell-env" ]; then
        USE_BLACKWELL=1
    fi
done

if [ "$USE_BLACKWELL" == "1" ]; then
    export PATH="/home/simran/.conda/envs/slimllm_bw/bin:$PATH"
else
    export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

# All 13 sessions: train ∪ val ∪ test
ALL_SESSIONS=(
    # train
    2025-03-17_10-36-44 2025-03-17_10-51-58 2025-03-17_11-06-39
    2025-03-18_12-39-10
    2025-03-19_10-05-47 2025-03-19_10-20-35 2025-03-19_10-37-56 2025-03-19_11-10-13
    2025-03-20_09-31-56 2025-03-20_09-46-30 2025-03-20_10-03-00
    # val
    2025-03-21_11-21-42 2025-03-25_13-23-42
    # test
    2025-03-25_13-39-06 2025-03-26_11-03-04
)

echo "=== Step 1: BL-7 dense embedding precompute ==="
echo "Sessions: ${#ALL_SESSIONS[@]}"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -u sensor_encoder_bl7.py \
    --ckpt models/bl7/bl7_best.pt \
    --sessions "${ALL_SESSIONS[@]}" \
    --output-dir sensor_embeddings_bl7/ \
    --batch-size 64 \
    --device cuda
PRECOMPUTE_RC=$?

if [ "$PRECOMPUTE_RC" != "0" ]; then
    echo "ERROR: precompute failed with exit $PRECOMPUTE_RC; aborting." >&2
    exit $PRECOMPUTE_RC
fi

echo ""
echo "=== Step 2: Submit 5-fold AR-modern + BL-7 training ==="
bash submit_ar_modern_bl7_5fold.sh
SUBMIT_RC=$?
exit $SUBMIT_RC
