#!/bin/bash
#SBATCH --job-name=vid_analysis
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

# Phases 2, 3, 4, 5, 6: Alignment → Prediction → Anomaly → LOSO → Report
# Run after all feature extraction (tiers 0-3) is complete.
# CPU-only. 16 cores for RF parallelism.

set -euo pipefail
echo "Host: $(hostname)"
echo "Start: $(date)"

export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
PYTHON=/home/simran/.conda/envs/slimllm/bin/python3
REPO=/home/simran/allspark-data-exploration/CPS-Debugger
cd "$REPO"

echo "=== Phase 2: Temporal Alignment ==="
$PYTHON -u scripts/video_assessment/p2_alignment.py

echo "=== Phase 3: Prediction Ladder ==="
$PYTHON -u scripts/video_assessment/p3_prediction.py

echo "=== Phase 4: Anomaly Detection ==="
$PYTHON -u scripts/video_assessment/p4_anomaly.py

echo "=== Phase 5: LOSO ==="
$PYTHON -u scripts/video_assessment/p5_loso.py

echo "=== Phase 6: Report ==="
$PYTHON -u scripts/video_assessment/p6_report.py

echo "All phases complete: $(date)"
