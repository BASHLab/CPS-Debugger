#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu-6-[01-20]
#SBATCH --output=models/slurm_bl7_w4_pre_%j.log

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"
export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p models results
echo "Wave-4 pre-flight: state-0/1/2 operand-2 decomposition (BL-7 v1 + BL-2)"
nvidia-smi || true

python3 -u bl7_w4_pre_state_decomp.py --device cuda
RC=$?
echo "Exit code: $RC"
exit $RC
