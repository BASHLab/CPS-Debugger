#!/bin/bash
# Submit 15 parallel full-vocab build jobs + 1 collect job
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p train_data_full

EXTRACTED="/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted"
TOKEN_MAP="${SCRIPT_DIR}/train_data_full/auto_token_mapping.json"
OUTPUT_DIR="${SCRIPT_DIR}/train_data_full"
OLD_DATA_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_trace_eda/train_data"

# Verify token mapping exists
if [ ! -f "$TOKEN_MAP" ]; then
    echo "ERROR: Token mapping not found: $TOKEN_MAP"
    echo "Run extract_wasm_edges.py first."
    exit 1
fi

SESSIONS=(
    "2025-03-17_10-36-44" "2025-03-17_10-51-58" "2025-03-17_11-06-39"
    "2025-03-18_12-39-10"
    "2025-03-19_10-05-47" "2025-03-19_10-20-35" "2025-03-19_10-37-56"
    "2025-03-19_11-10-13"
    "2025-03-20_09-31-56" "2025-03-20_09-46-30" "2025-03-20_10-03-00"
    "2025-03-21_11-21-42"
    "2025-03-25_13-23-42" "2025-03-25_13-39-06"
    "2025-03-26_11-03-04"
)

# Build session list string for collect job
SESS_ARGS=""
for SESS in "${SESSIONS[@]}"; do
    SESS_ARGS="${SESS_ARGS} ${SESS}"
done

JOB_IDS=""
for SESS in "${SESSIONS[@]}"; do
    JID=$(sbatch --parsable --job-name="bfull_${SESS}" run_build_session_full.sh "$SESS")
    echo "Submitted bfull_${SESS}: $JID"
    if [ -n "$JOB_IDS" ]; then
        JOB_IDS="${JOB_IDS}:${JID}"
    else
        JOB_IDS="${JID}"
    fi
done

# Collect job after all builds complete
COLLECT_JID=$(sbatch --parsable \
    --dependency=afterok:${JOB_IDS} \
    --job-name=full_collect \
    --partition=short \
    --cpus-per-task=4 \
    --mem=48G \
    --time=01:00:00 \
    --output=train_data_full/slurm_%x_%j.log \
    --wrap="cd ${SCRIPT_DIR} && export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH && python3 aspk_dataset_collect_full.py --data-dir ${OUTPUT_DIR} --token-mapping ${TOKEN_MAP} --sessions ${SESS_ARGS} --old-data-dir ${OLD_DATA_DIR}")

echo ""
echo "Submitted ${#SESSIONS[@]} build jobs + collect job ($COLLECT_JID)"
echo "Dependency chain: collect afterok:${JOB_IDS}"
echo ""
echo "Monitor: squeue -u \$USER"
