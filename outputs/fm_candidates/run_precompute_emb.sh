#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH --output=sensor_embeddings/slurm_emb_%j.log

# Precompute sensor embeddings for one or more sessions.
# Designed to be launched in parallel (one job per session).
#
# Usage:
#   sbatch run_precompute_emb.sh 2025-03-17_10-36-44

set -e
SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"
export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"

mkdir -p sensor_embeddings

if [ $# -eq 0 ]; then
    echo "ERROR: No sessions specified. Usage: sbatch run_precompute_emb.sh <session1> [session2] ..."
    exit 1
fi

echo "Precomputing embeddings for $# session(s): $@"
echo "Device: $(python3 -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
nvidia-smi || true

python3 sensor_encoder.py \
    --sessions "$@" \
    --output-dir sensor_embeddings/ \
    --data-dir train_data_full/ \
    --batch-size 32 \
    --seed 42

echo "Exit code: $?"
