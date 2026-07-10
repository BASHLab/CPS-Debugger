#!/bin/bash
#SBATCH --partition=short
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=train_data_full/slurm_%x_%j.log

# Usage: sbatch --job-name=bfull_<session> run_build_session_full.sh <session_id>
SESSION=$1
if [ -z "$SESSION" ]; then
    echo "ERROR: provide session id as argument"
    exit 1
fi

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
EXTRACTED="/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted"
TOKEN_MAP="${SCRIPT_DIR}/train_data_full/auto_token_mapping.json"
OUTPUT_DIR="${SCRIPT_DIR}/train_data_full"

cd "$SCRIPT_DIR"
export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"

echo "Building full-vocab parquet for session: $SESSION"
python3 aspk_build_session_full.py \
    --session "$SESSION" \
    --extracted-dir "$EXTRACTED" \
    --token-mapping "$TOKEN_MAP" \
    --output-dir "$OUTPUT_DIR" \
    --workers 8
echo "Exit code: $?"
