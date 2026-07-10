#!/bin/bash
#SBATCH --job-name=cps_phase2A
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

# Phase A: State-stratified R² decomposition.
# Runs 4 separate CV experiments + FWL residualisation + leakage check.
# Most expensive: per-state RF on 130k BALANCE windows.

set -euo pipefail
echo "Host: $(hostname)"
echo "Start: $(date)"

PYTHON=/home/simran/.conda/envs/slimllm/bin/python3
REPO=/home/simran/allspark-data-exploration/CPS-Debugger
cd "$REPO"

$PYTHON -u scripts/phase2_A_stratified_analysis.py

echo "Done: $(date)"
