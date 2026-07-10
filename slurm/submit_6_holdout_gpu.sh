#!/bin/bash
set -euo pipefail

ROOT=/home/simran/allspark-data-exploration/CPS-Debugger
SCRIPT=$ROOT/slurm/run_holdout_mlp_gpu.sbatch

if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing sbatch script: $SCRIPT"
  exit 1
fi

RUNS=(
  "2025-03-13_09-23-44"
  "2025-03-13_11-12-19"
  "2025-03-13_13-31-50"
  "2025-03-13_14-32-27"
  "2025-03-17_10-06-14"
  "2025-03-17_10-21-19"
)

echo "Submitting ${#RUNS[@]} holdout GPU jobs..."
for holdout in "${RUNS[@]}"; do
  job_out=$(sbatch --export=ALL,TEST_RUN="$holdout" "$SCRIPT")
  echo "$holdout -> $job_out"
done

