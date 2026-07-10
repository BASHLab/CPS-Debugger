#!/bin/bash
#SBATCH --job-name=cps_exp06_cross_run
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

# Experiment 06: Leave-one-run-out cross-run generalization.
# 6 folds × 2 RF models (all features + physical-only) on ~134k train windows each.
# Needs 64G to avoid OOM from RF multi-output trees.

set -euo pipefail

echo "Host: $(hostname)"
echo "Start: $(date)"

PYTHON=/home/simran/.conda/envs/slimllm/bin/python3
REPO=/home/simran/allspark-data-exploration/CPS-Debugger
cd "$REPO"

$PYTHON -u scripts/06_cross_run_generalization.py

echo "Done: $(date)"
