#!/bin/bash
# Submit 15 parallel pruned-vocab build jobs.
# Uses train_data_full/pruned_token_mapping.json and writes to train_data_pruned/.
# Existing full-vocab parquets in train_data_full/ are preserved for ablation.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p train_data_pruned

TOKEN_MAP="${SCRIPT_DIR}/train_data_full/pruned_token_mapping.json"

if [ ! -f "$TOKEN_MAP" ]; then
    echo "ERROR: Pruned token mapping not found: $TOKEN_MAP"
    echo "Run: python3 extract_wasm_edges.py ... --prune"
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

JOB_IDS=""
for SESS in "${SESSIONS[@]}"; do
    # Skip if already built
    if [ -f "train_data_pruned/${SESS}.parquet" ]; then
        echo "Skipping ${SESS} (already built)"
        continue
    fi
    JID=$(sbatch --parsable --job-name="bprn_${SESS}" run_build_session_pruned.sh "$SESS")
    echo "Submitted bprn_${SESS}: $JID"
    if [ -n "$JOB_IDS" ]; then
        JOB_IDS="${JOB_IDS}:${JID}"
    else
        JOB_IDS="${JID}"
    fi
done

echo ""
echo "Submitted build jobs for pruned vocab"
echo "Monitor: squeue -u \$USER"
