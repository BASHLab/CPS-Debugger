#!/bin/bash
#SBATCH --partition=short
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=train_data/slurm_%x_%j.log

# Usage: sbatch --job-name=build_<session> run_build_session.sh <session_id>
SESSION=$1
if [ -z "$SESSION" ]; then
    echo "ERROR: provide session id as argument"
    exit 1
fi

cd /home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_trace_eda
export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"

echo "Building parquet for session: $SESSION"
python3 aspk_build_session.py --session "$SESSION" --workers 8
echo "Exit code: $?"
