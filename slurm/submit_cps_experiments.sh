#!/bin/bash
# submit_cps_experiments.sh
#
# Submits CPS trace-recovery experiments to SLURM in parallel,
# then a summary job after both complete.
#
# Usage:
#   bash slurm/submit_cps_experiments.sh [--dry-run]

set -euo pipefail

REPO=/home/simran/allspark-data-exploration/CPS-Debugger
SLURM="$REPO/slurm"

DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY=1; echo "[DRY RUN]"
fi

submit() {
  local script="$1"; shift
  if [[ $DRY -eq 1 ]]; then
    echo "[DRY] sbatch $* $script"
    echo "99999"
  else
    sbatch --parsable "$@" "$script"
  fi
}

echo "Submitting exp01 (branch count regression, 16 CPUs, 48G, 4h)..."
JOB01=$(submit "$SLURM/run_exp01_branch_regression.sh")
echo "  Job ID: $JOB01"

echo "Submitting exp06 (cross-run generalization, 16 CPUs, 64G, 6h)..."
JOB06=$(submit "$SLURM/run_exp06_cross_run.sh")
echo "  Job ID: $JOB06"

echo "Submitting summary job (after both complete)..."
JOB_SUM=$(submit "$SLURM/run_exp_summary.sh" \
  --dependency="afterok:${JOB01}:${JOB06}")
echo "  Job ID: $JOB_SUM"

echo ""
echo "Submitted 3 jobs: exp01=$JOB01  exp06=$JOB06  summary=$JOB_SUM"
echo "Monitor: squeue -u \$(whoami)"
echo "Logs:    $REPO/outputs/slurm/"
