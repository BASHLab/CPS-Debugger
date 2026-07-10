#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH --output=sensor_embeddings_moment/slurm_extract_%j.log

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
mkdir -p "$SCRIPT_DIR/sensor_embeddings_moment"
cd "$SCRIPT_DIR"
export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"

SESSION="$1"
shift 2>/dev/null || true
if [ -z "$SESSION" ]; then
    echo "ERROR: --session required as first positional"
    exit 1
fi

python3 -u sensor_encoder_moment.py --sessions "$SESSION" "$@"
echo "Exit code: $?"
