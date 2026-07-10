#!/bin/bash
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=trace_vocab_inv
#SBATCH --output=train_data/slurm_%x_%j.log

cd /home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_trace_eda
export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"

echo "Starting trace vocabulary inventory..."
python3 trace_vocab_inventory.py
echo "Exit code: $?"
