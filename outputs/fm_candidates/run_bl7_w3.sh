#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH --output=models/slurm_bl7_w3_%j.log

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"
export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p models results
echo "BL-7 W3 operand-2 concentration probe"
nvidia-smi || true

python3 -u bl7_w3_operand2_concentration.py --device cuda
RC=$?

echo "Exit code: $RC"
exit $RC
