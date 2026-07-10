#!/bin/bash
# Chunked test eval for BL-3b (LoRA V-JEPA). 4 chunks per fold + 1 merge.
# V-JEPA forward dominates inference, so smaller batch + any-GPU; 6h cap.

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")
N_SAMPLES=30000
N_CHUNKS=4
CHUNK_SIZE=$((N_SAMPLES / N_CHUNKS))

JOBS=""
for fold_idx in 0 1 2 3 4; do
    fold_dir="$SCRIPT_DIR/models/bl3b_fold_${fold_idx}"
    result_dir="$SCRIPT_DIR/results/bl3b_fold_${fold_idx}"
    mkdir -p "$result_dir"
    chunk_jobs=""
    for chunk_id in 0 1 2 3; do
        start=$((chunk_id * CHUNK_SIZE)); end=$((start + CHUNK_SIZE))
        JOB=$(sbatch --parsable \
            --job-name="bl3b_f${fold_idx}_test_c${chunk_id}" \
            --partition=short --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=06:00:00 \
            --exclude=gpu-6-[01-20] \
            --output="$result_dir/slurm_bl3b_test_c${chunk_id}_%j.log" \
            --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
CKPT=$fold_dir/bl3b_best.pt
echo \"Chunk $chunk_id [\$start, \$end) of ${N_SAMPLES}\"
python3 -u bl3b_eval.py --checkpoint \$CKPT --split test --sessions ${TEST_SESSIONS[*]} --device cuda --n-samples $N_SAMPLES --n-samples-start $start --n-samples-end $end --num-workers 0 --save-raw $result_dir/bl3b_eval_test_chunk_${chunk_id}.npz --result-dir $result_dir
")
        chunk_jobs="${chunk_jobs}:${JOB}"
        echo "  bl3b f${fold_idx} c${chunk_id} -> $JOB"
        JOBS="$JOBS $JOB"
    done
    MERGE=$(sbatch --parsable \
        --job-name="bl3b_f${fold_idx}_merge" \
        --partition=short --cpus-per-task=2 --mem=32G --time=01:00:00 \
        --dependency="afterok${chunk_jobs}" \
        --output="$result_dir/slurm_bl3b_merge_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export TOKENIZERS_PARALLELISM=false
python3 -u merge_chunks.py --chunks $result_dir/bl3b_eval_test_chunk_*.npz --result-dir $result_dir --name bl3b --split test
")
    echo "  bl3b f${fold_idx} merge -> $MERGE"
    JOBS="$JOBS $MERGE"
done
echo ""
echo "All BL-3b chunked eval jobs:$JOBS"
