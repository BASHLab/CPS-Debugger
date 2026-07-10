#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH --output=results/slurm_eval_%j.log

# Usage: sbatch run_eval.sh <checkpoint> [extra args to evaluate.py]
# Default: evaluates both val and test splits.

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"
export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p results

CHECKPOINT="${1:-models/mdlm_best.pt}"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    exit 1
fi

shift 2>/dev/null || true

echo "=== Evaluating VAL split ==="
python3 -u evaluate.py --checkpoint "$CHECKPOINT" --split val --device cuda "$@"

echo "=== Evaluating TEST split ==="
python3 -u evaluate.py --checkpoint "$CHECKPOINT" --split test --device cuda "$@"

echo "Exit code: $?"
