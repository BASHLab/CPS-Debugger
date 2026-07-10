#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH --output=models/slurm_bl7_diagnostics_%j.log

# BL-7 within-state-1 gap — pre-research diagnostics (Diag 1-3).
# Diag 1 is the HARD GATE. Diag 2/3 are CPU-bound (sklearn/lightgbm).
# Usage:
#   sbatch run_bl7_diagnostics.sh all
#   sbatch run_bl7_diagnostics.sh diag1

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"
export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p models results

echo "BL-7 diagnostics: ${@:-all}"
nvidia-smi || true

python3 -u bl7_diagnostics.py "${@:-all}" --device cuda
RC=$?

echo "Exit code: $RC"
exit $RC
