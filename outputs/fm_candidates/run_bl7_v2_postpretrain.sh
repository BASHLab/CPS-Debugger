#!/bin/bash
# Auto-pipeline that runs after BL-7-v2 (Tier 2 contrastive) pretrain.
#   1. Precompute v2 dense embeddings → sensor_embeddings_bl7_v2/
#   2. Submit 5-fold AR-modern training + stratified-100K eval on v2 embeddings
#      (Cell B of the 2x2 factorial: prefix decoder + Tier 2 contrastive pretrain)
#
# Submit with:
#   sbatch --gres=gpu:H100:1 --time=08:00:00 \
#          --dependency=afterok:<bl7_v2_pretrain_jobid> \
#          --output=results/bl7_v2_postpretrain_%j.log \
#          run_bl7_v2_postpretrain.sh

#SBATCH --job-name=bl7_v2_postpretrain
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

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

echo "=== Step 1: BL-7-v2 dense embedding precompute ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -u sensor_encoder_bl7.py \
    --ckpt models/bl7_v2/bl7/bl7_best.pt \
    --sessions "${ALL_SESSIONS[@]}" \
    --output-dir sensor_embeddings_bl7_v2/ \
    --batch-size 64 \
    --device cuda
PRECOMPUTE_RC=$?
[ "$PRECOMPUTE_RC" != "0" ] && { echo "ERROR: v2 precompute failed exit $PRECOMPUTE_RC"; exit $PRECOMPUTE_RC; }

echo ""
echo "=== Step 2: Submit Cell B (prefix decoder + Tier 2 contrastive pretrain) ==="
EMB_DIR="$SCRIPT_DIR/sensor_embeddings_bl7_v2" \
    VARIANT_NAME="ar_modern_bl7_v2" \
    EXTRA_TRAIN_ARGS="" \
    bash submit_ar_modern_bl7_5fold.sh
RC_B=$?

echo ""
echo "=== Step 3: Submit Cell C (cross-attn decoder + Tier 2 contrastive pretrain) ==="
EMB_DIR="$SCRIPT_DIR/sensor_embeddings_bl7_v2" \
    VARIANT_NAME="ar_modern_bl7_v2_cross" \
    EXTRA_TRAIN_ARGS="--use-cross-attn --n-sensor-tokens 4 --cond-dropout-p 0.10" \
    bash submit_ar_modern_bl7_5fold.sh
RC_C=$?

[ "$RC_B" != "0" ] || [ "$RC_C" != "0" ] && exit 1
exit 0
