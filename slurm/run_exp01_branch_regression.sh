#!/bin/bash
#SBATCH --job-name=cps_exp01_branch_reg
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

# Experiment 01: Branch count regression (physical+syslog → branch execution counts).
# Re-runs with enough time to complete all figures including state-level R² analysis.

set -euo pipefail

echo "Host: $(hostname)"
echo "Start: $(date)"

PYTHON=/home/simran/.conda/envs/slimllm/bin/python3
REPO=/home/simran/allspark-data-exploration/CPS-Debugger
cd "$REPO"

$PYTHON -u scripts/01_branch_count_regression.py

echo "Done: $(date)"
