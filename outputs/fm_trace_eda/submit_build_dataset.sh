#!/bin/bash
# Submit 15 parallel build jobs + 1 collect job
set -e
cd /home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_trace_eda
mkdir -p train_data

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
    JID=$(sbatch --parsable --job-name="build_${SESS}" run_build_session.sh "$SESS")
    echo "Submitted build_${SESS}: $JID"
    if [ -n "$JOB_IDS" ]; then
        JOB_IDS="${JOB_IDS}:${JID}"
    else
        JOB_IDS="${JID}"
    fi
done

# Collect job after all builds complete
COLLECT_JID=$(sbatch --parsable \
    --dependency=afterok:${JOB_IDS} \
    --job-name=dataset_collect \
    --partition=short \
    --cpus-per-task=4 \
    --mem=32G \
    --time=00:30:00 \
    --output=train_data/slurm_%x_%j.log \
    --wrap="cd /home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_trace_eda && export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH && python3 aspk_dataset_collect.py")

echo ""
echo "Submitted $((${#SESSIONS[@]})) build jobs + collect job ($COLLECT_JID)"
echo "Dependency chain: collect afterok:${JOB_IDS}"
echo ""
echo "Monitor: squeue -u \$USER"
