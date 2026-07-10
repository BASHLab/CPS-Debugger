#!/bin/bash
# Chunked test-eval pipeline:
#  - Each (variant, fold) test eval is split into N_CHUNKS sub-jobs over the
#    deterministic permutation slice [start, end). Each chunk dumps raw
#    sequences via `--save-raw`.
#  - A merge job depends (afterok) on all chunks for that (variant, fold)
#    and concatenates them through compute_generation_metrics → final JSON.
#
# Total per (variant, fold): N_CHUNKS chunks + 1 merge.
#
# MDLM stays H100-pinned (sampling is the slow part; needs memory headroom).
# AR variants + baselines run on any non-Blackwell GPU.

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")
N_SAMPLES=30000
N_CHUNKS=4
CHUNK_SIZE=$((N_SAMPLES / N_CHUNKS))

# Argument: variant ar_modern / bl3 / bl4 / bl5 / bl6 (plus "all" pseudo)
# folds_arg: optional space-separated list (default 0 1 2 3 4)
VARIANT="${1:?Usage: $0 <variant> [folds...]}"
shift
FOLDS=("$@")
if [ ${#FOLDS[@]} -eq 0 ]; then
    FOLDS=(0 1 2 3 4)
fi

# Hardware policy
if [ "$VARIANT" = "mdlm" ]; then
    GRES="gpu:H100:1"
    MEM="64G"
    TIME="06:00:00"
else
    GRES="gpu:1"
    MEM="64G"
    TIME="04:00:00"
fi

# CKPT name pattern
case "$VARIANT" in
    mdlm)        CKPT_NAME="bl2_best.pt"; FOLD_PREFIX="fold" ;;
    ar_modern)   CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="ar_modern_fold" ;;
    bl3|bl4|bl5|bl6|bl6b)
                 CKPT_NAME="${VARIANT}_best.pt"; FOLD_PREFIX="${VARIANT}_fold" ;;
    *) echo "ERROR: unknown variant $VARIANT" >&2; exit 1 ;;
esac

JOBS=""
for fold_idx in "${FOLDS[@]}"; do
    fold_dir="$SCRIPT_DIR/models/${FOLD_PREFIX}_${fold_idx}"
    case "$VARIANT" in
        mdlm) result_dir="$SCRIPT_DIR/results/mdlm_fold_${fold_idx}" ;;
        ar_modern) result_dir="$SCRIPT_DIR/results/ar_modern_fold_${fold_idx}" ;;
        *) result_dir="$SCRIPT_DIR/results/${VARIANT}_fold_${fold_idx}" ;;
    esac
    mkdir -p "$result_dir"

    chunk_jobs=""
    for chunk_id in $(seq 0 $((N_CHUNKS - 1))); do
        start=$((chunk_id * CHUNK_SIZE))
        end=$((start + CHUNK_SIZE))
        raw_path="$result_dir/${VARIANT}_eval_test_chunk_${chunk_id}.npz"

        # MDLM-specific extra arg (sampling steps); other variants ignore
        extra_args=""
        [ "$VARIANT" = "mdlm" ] && extra_args="--sampling-steps 200"

        JOB=$(sbatch --parsable \
            --job-name="${VARIANT}_f${fold_idx}_test_c${chunk_id}" \
            --partition=short \
            --gres="$GRES" \
            --cpus-per-task=4 \
            --mem="$MEM" \
            --time="$TIME" \
            --exclude=gpu-6-[01-20] \
            --output="$result_dir/slurm_${VARIANT}_test_c${chunk_id}_%j.log" \
            --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
CKPT=$fold_dir/$CKPT_NAME
if [ ! -f \$CKPT ]; then
    CKPT=\$(ls -t $fold_dir/${VARIANT}_final_*.pt 2>/dev/null | head -1)
fi
echo \"Chunk $chunk_id [\$start, \$end) of ${N_SAMPLES}\"
python3 -u evaluate.py \
    --checkpoint \$CKPT \
    --split test \
    --sessions ${TEST_SESSIONS[*]} \
    --device cuda \
    --n-samples $N_SAMPLES \
    --n-samples-start $start \
    --n-samples-end $end \
    --num-workers 0 \
    --save-raw $raw_path \
    --result-dir $result_dir $extra_args
")
        chunk_jobs="${chunk_jobs}:${JOB}"
        echo "  $VARIANT f$fold_idx chunk $chunk_id [$start,$end) -> $JOB"
        JOBS="$JOBS $JOB"
    done

    # Merge job
    MERGE=$(sbatch --parsable \
        --job-name="${VARIANT}_f${fold_idx}_merge" \
        --partition=short \
        --cpus-per-task=2 \
        --mem=32G \
        --time=01:00:00 \
        --dependency="afterok${chunk_jobs}" \
        --output="$result_dir/slurm_${VARIANT}_merge_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export TOKENIZERS_PARALLELISM=false
python3 -u merge_chunks.py \
    --chunks ${result_dir}/${VARIANT}_eval_test_chunk_*.npz \
    --result-dir $result_dir \
    --name $VARIANT --split test
")
    echo "  $VARIANT f$fold_idx merge -> $MERGE (after$chunk_jobs)"
    JOBS="$JOBS $MERGE"
done

echo ""
echo "Submitted ${VARIANT} chunked evals:$JOBS"
