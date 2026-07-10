#!/bin/bash
# Fan out MOMENT per-tick embedding extraction across all 15 sessions.
# Run AFTER finetune_moment.py has produced models/moment_finetuned/moment_best.pt.

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

ALL_SESSIONS=(
    "2025-03-17_10-36-44" "2025-03-17_10-51-58" "2025-03-17_11-06-39"
    "2025-03-18_12-39-10" "2025-03-19_10-05-47" "2025-03-19_10-20-35"
    "2025-03-19_10-37-56" "2025-03-19_11-10-13" "2025-03-20_09-31-56"
    "2025-03-20_09-46-30" "2025-03-20_10-03-00"
    "2025-03-21_11-21-42" "2025-03-25_13-23-42"
    "2025-03-25_13-39-06" "2025-03-26_11-03-04"
)

JOBS=""
for sess in "${ALL_SESSIONS[@]}"; do
    JOB=$(sbatch --parsable \
        --job-name="moment_${sess}" \
        run_sensor_encoder_moment.sh "$sess")
    echo "  $sess -> $JOB"
    JOBS="$JOBS $JOB"
done
echo ""
echo "All MOMENT extraction jobs submitted:$JOBS"
