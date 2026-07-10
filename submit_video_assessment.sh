#!/bin/bash
# submit_video_assessment.sh — Master submission for the Video Value Assessment sprint.
#
# Usage:
#   bash submit_video_assessment.sh           # submit everything
#   bash submit_video_assessment.sh --tier 0  # only Tier 0 (CPU optical flow)
#   bash submit_video_assessment.sh --analysis-only  # skip extraction, run phases 2-6
#
# Dependency chain:
#   Phase 0 (CPU, ~2min)
#     → Tier 0 (CPU, ~2h all sessions)
#     → Tiers 1,2,3 (GPU, ~2-4h per session, parallel)
#       → Tier 2 PCA (CPU, after all tier2 raw extraction)
#       → Tier 3 PCA (CPU, after all tier3 raw extraction)
#         → Phases 2-6 (CPU, ~4h)

set -euo pipefail

REPO=/home/simran/allspark-data-exploration/CPS-Debugger
cd "$REPO"

# Parse args
TIER_ONLY=""
ANALYSIS_ONLY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier) TIER_ONLY="$2"; shift 2 ;;
        --analysis-only) ANALYSIS_ONLY=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Load video sessions from params or detect
PYTHON=/home/simran/.conda/envs/slimllm/bin/python3
PARAMS_FILE="$REPO/outputs/video_assessment/dataset_params.json"

if [ ! -f "$PARAMS_FILE" ]; then
    echo "Running Phase 0 reconnaissance first..."
    export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
    $PYTHON -u scripts/video_assessment/p0_recon.py
fi

# Read session list
VIDEO_SESSIONS=$($PYTHON -c "
import json
p = json.load(open('$PARAMS_FILE'))
sessions = p.get('full_video_sessions', p.get('video_sessions', []))
print(' '.join(sessions))
")
echo "Video sessions: $VIDEO_SESSIONS"

# ── Tier 0 ────────────────────────────────────────────────────────────────────
if [ -z "$TIER_ONLY" ] || [ "$TIER_ONLY" = "0" ]; then
    if ! $ANALYSIS_ONLY; then
        echo "Submitting Tier 0 (optical flow, all sessions)..."
        JOB_T0=$(sbatch --parsable slurm/video_tier0_all.sh)
        echo "  Tier 0 job: $JOB_T0"
    fi
fi

# ── Tier 1 ────────────────────────────────────────────────────────────────────
T1_JOBS=""
if [ -z "$TIER_ONLY" ] || [ "$TIER_ONLY" = "1" ]; then
    if ! $ANALYSIS_ONLY; then
        echo "Submitting Tier 1 (CoTracker3, one job per session)..."
        for RUN_ID in $VIDEO_SESSIONS; do
            JOB=$(sbatch --parsable --export=RUN_ID="$RUN_ID" slurm/video_tier1_session.sh)
            T1_JOBS="$T1_JOBS:$JOB"
            echo "  Tier 1 [$RUN_ID]: job $JOB"
        done
    fi
fi

# ── Tier 2 ────────────────────────────────────────────────────────────────────
T2_RAW_JOBS=""
if [ -z "$TIER_ONLY" ] || [ "$TIER_ONLY" = "2" ]; then
    if ! $ANALYSIS_ONLY; then
        echo "Submitting Tier 2 raw extraction (DINOv2, one job per session)..."
        for RUN_ID in $VIDEO_SESSIONS; do
            JOB=$(sbatch --parsable --export=RUN_ID="$RUN_ID" slurm/video_tier2_session.sh)
            T2_RAW_JOBS="$T2_RAW_JOBS:$JOB"
            echo "  Tier 2 raw [$RUN_ID]: job $JOB"
        done

        # PCA step (after all raw extractions)
        T2_RAW_DEPS="${T2_RAW_JOBS#:}"  # remove leading colon
        JOB_T2_PCA=$(sbatch --parsable \
            --dependency="afterok:${T2_RAW_DEPS//:/ :}" \
            slurm/video_tier2_pca.sh 2>/dev/null || \
            sbatch --parsable slurm/video_tier2_pca.sh)
        echo "  Tier 2 PCA: job $JOB_T2_PCA (after $T2_RAW_DEPS)"
    fi
fi

# ── Tier 3 ────────────────────────────────────────────────────────────────────
T3_RAW_JOBS=""
if [ -z "$TIER_ONLY" ] || [ "$TIER_ONLY" = "3" ]; then
    if ! $ANALYSIS_ONLY; then
        echo "Submitting Tier 3 (V-JEPA 2, one job per session on L40S)..."
        for RUN_ID in $VIDEO_SESSIONS; do
            JOB=$(sbatch --parsable --export=RUN_ID="$RUN_ID" slurm/video_tier3_session.sh \
                  2>/dev/null || \
                  # Fallback: any GPU if L40S unavailable
                  sbatch --parsable --export=RUN_ID="$RUN_ID" \
                      --gres=gpu:1 slurm/video_tier3_session.sh)
            T3_RAW_JOBS="$T3_RAW_JOBS:$JOB"
            echo "  Tier 3 [$RUN_ID]: job $JOB"
        done

        # PCA (run inline after all T3 raw done — submit here with dependency)
        T3_RAW_DEPS="${T3_RAW_JOBS#:}"
        echo "  Tier 3 PCA will run in phases_2345 job after extraction"
    fi
fi

# ── Phases 2-6 ────────────────────────────────────────────────────────────────
echo ""
echo "Submitting Phases 2-6 (alignment, prediction, anomaly, LOSO, report)..."

# Build dependency list from all extraction jobs
ALL_EXTRACT_JOBS=""
[ -n "$JOB_T0" ]     && ALL_EXTRACT_JOBS="$ALL_EXTRACT_JOBS:$JOB_T0"
[ -n "$T1_JOBS" ]    && ALL_EXTRACT_JOBS="$ALL_EXTRACT_JOBS$T1_JOBS"
[ -n "$JOB_T2_PCA" ] && ALL_EXTRACT_JOBS="$ALL_EXTRACT_JOBS:$JOB_T2_PCA"
[ -n "$T3_RAW_JOBS" ] && ALL_EXTRACT_JOBS="$ALL_EXTRACT_JOBS$T3_RAW_JOBS"

if [ -n "$ALL_EXTRACT_JOBS" ] && ! $ANALYSIS_ONLY; then
    # First fit tier3 PCA inline in the phases job
    DEPS="${ALL_EXTRACT_JOBS#:}"
    JOB_ANALYSIS=$(sbatch --parsable \
        --dependency="afterany:${DEPS//:/ :}" \
        slurm/video_phases_2345.sh)
else
    JOB_ANALYSIS=$(sbatch --parsable slurm/video_phases_2345.sh)
fi
echo "  Phases 2-6 job: $JOB_ANALYSIS"

echo ""
echo "=== Submitted job summary ==="
echo "  Tier 0 (optical flow):    ${JOB_T0:-skipped}"
echo "  Tier 1 (CoTracker3):      ${T1_JOBS:1:-skipped}"
echo "  Tier 2 raw (DINOv2):      ${T2_RAW_JOBS:1:-skipped}"
echo "  Tier 2 PCA:               ${JOB_T2_PCA:-skipped}"
echo "  Tier 3 (V-JEPA 2):        ${T3_RAW_JOBS:1:-skipped}"
echo "  Phases 2-6 analysis:      $JOB_ANALYSIS"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: outputs/video_assessment/VIDEO_VALUE_SUMMARY.md"
