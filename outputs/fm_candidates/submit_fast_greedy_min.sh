#!/bin/bash
# Deadline-fast greedy-only eval on the minorities-complete first 25K of the
# stratified-100K list (ALL state-0 + ALL state-2 + ~4.4K state-1). 6 chunks
# of ~4.2K each per variant (12 total fits the QOS cap, no queue), greedy-only
# (skip the sampled pass). Decisive per-state NED (s0/s2 complete) for the
# contrastive (v2) and HuBERT (hub) K8 encoders, before 2PM.
set -euo pipefail
SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")
SUBSET=25000
NCH=6
CHUNK=$(( (SUBSET + NCH - 1) / NCH ))   # ~4167

for VARIANT in ar_modern_bl7_v2_w4000_k8 ar_modern_bl7_hub_w4000_k8; do
    FOLD_DIR="$SCRIPT_DIR/models/${VARIANT}_fold_0"
    RD="$SCRIPT_DIR/results/${VARIANT}_strat100k_fold_0"
    EMB="$SCRIPT_DIR/sensor_embeddings_${VARIANT}_fold_0"
    CKPT="$FOLD_DIR/ar_modern_best.pt"
    mkdir -p "$RD"
    test -f "$CKPT" || { echo "ERROR: $CKPT missing" >&2; exit 1; }
    chunk_deps=""
    for c in $(seq 0 $((NCH-1))); do
        start=$((c*CHUNK)); end=$((start+CHUNK)); [ $end -gt $SUBSET ] && end=$SUBSET
        J=$(sbatch --parsable \
            --partition=short --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=00:40:00 \
            --exclude=gpu-6-[01-20] \
            --job-name="${VARIANT}_fastmin_c${c}" \
            --output="$RD/slurm_fastmin_c${c}_%j.log" \
            --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -u evaluate.py --checkpoint $CKPT --split test --sessions ${TEST_SESSIONS[*]} \
    --device cuda --stratified-target 100000 --n-samples-start $start --n-samples-end $end \
    --num-workers 0 --greedy-only --emb-dir $EMB \
    --save-raw $RD/${VARIANT}_fastmin_eval_test_chunk_${c}.npz --result-dir $RD")
        chunk_deps="${chunk_deps}:${J}"
        echo "  $VARIANT fastmin c$c [$start,$end) -> $J"
    done
    M=$(sbatch --parsable --dependency="afterok${chunk_deps}" \
        --partition=short --cpus-per-task=2 --mem=32G --time=00:30:00 \
        --output="$RD/slurm_fastmin_merge_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
python3 -u merge_chunks.py --chunks $RD/${VARIANT}_fastmin_eval_test_chunk_*.npz \
    --result-dir $RD --name ${VARIANT}_fastmin --split test")
    echo "  $VARIANT fastmin merge -> $M (after$chunk_deps)"
done
echo "Fast greedy-only minorities eval submitted."
