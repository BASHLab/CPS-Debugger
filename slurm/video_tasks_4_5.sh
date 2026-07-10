#!/bin/bash
#SBATCH --job-name=vid_tasks45
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

# Tasks 4, 5: LLM reasoning with visual evidence + code verification.
# Requires Claude API key in environment (ANTHROPIC_API_KEY).

set -euo pipefail
echo "Host: $(hostname)"
echo "Start: $(date)"

export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
PYTHON=/home/simran/.conda/envs/slimllm/bin/python3
REPO=/home/simran/allspark-data-exploration/CPS-Debugger
cd "$REPO"

echo "=== Task 4: LLM Reasoning ==="
$PYTHON -u scripts/video_assessment/task4_llm_reasoning.py

echo "=== Task 5: Code Verification ==="
$PYTHON -u scripts/video_assessment/task5_code_verification.py

echo "All tasks 4-5 complete: $(date)"
