#!/bin/bash
#SBATCH --job-name=vid_tasks123
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

# Tasks 1, 2, 3: State estimation, FSM classification, Anomaly detection.
# CPU-only. ~2-3 hours depending on RF forest sizes.

set -euo pipefail
echo "Host: $(hostname)"
echo "Start: $(date)"

export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
PYTHON=/home/simran/.conda/envs/slimllm/bin/python3
REPO=/home/simran/allspark-data-exploration/CPS-Debugger
cd "$REPO"

echo "=== Task 1: State Estimation ==="
$PYTHON -u scripts/video_assessment/task1_state_estimation.py

echo "=== Task 2: FSM Classification ==="
$PYTHON -u scripts/video_assessment/task2_fsm_classification.py

echo "=== Task 3: Anomaly Detection ==="
$PYTHON -u scripts/video_assessment/task3_anomaly_detection.py

echo "All tasks 1-3 complete: $(date)"
