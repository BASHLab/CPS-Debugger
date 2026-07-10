#!/bin/bash
# Submit stratified-100K test eval for one variant: 5 folds × N_CHUNKS chunks
# + 1 merge per fold. Stratified subset = ALL state-0 (~17.6K) + ALL state-2
# (~3K) + ~79K from state-1 (deterministic via EVAL_SUBSET_SEED).
#
# Usage: bash submit_stratified_100k.sh <variant> [folds...]
# AR variants (ar_modern/bl3/bl4/bl6) use 4 chunks of ~25K (any-GPU).
# MDLM uses 8 chunks of ~12.5K to stay under the 6h SLURM cap (H100).

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")
TARGET=100000

VARIANT="${1:?Usage: $0 <variant> [folds...]}"
shift
FOLDS=("$@")
[ ${#FOLDS[@]} -eq 0 ] && FOLDS=(0 1 2 3 4)

CHUNK_TIME="${CHUNK_TIME:-06:00:00}"

case "$VARIANT" in
    mdlm)                       N_CHUNKS=8;  GRES="gpu:H100:1"; CKPT_NAME="bl2_best.pt"; FOLD_PREFIX="fold"; EXTRA="--sampling-steps 200"; ENTRY="evaluate.py" ;;
    ar_modern)                  N_CHUNKS=4;  GRES="gpu:1";      CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="ar_modern_fold"; EXTRA=""; ENTRY="evaluate.py" ;;
    ar_modern_bl7)              N_CHUNKS=4;  GRES="gpu:1";      CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="ar_modern_bl7_fold"; EXTRA="--emb-dir $SCRIPT_DIR/sensor_embeddings_bl7"; ENTRY="evaluate.py" ;;
    # 2x2 factorial cells:
    ar_modern_bl7_v2)           N_CHUNKS=4;  GRES="gpu:1";      CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="ar_modern_bl7_v2_fold"; EXTRA="--emb-dir $SCRIPT_DIR/sensor_embeddings_bl7_v2"; ENTRY="evaluate.py" ;;
    ar_modern_bl7_cross)        N_CHUNKS=4;  GRES="gpu:1";      CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="ar_modern_bl7_cross_fold"; EXTRA="--emb-dir $SCRIPT_DIR/sensor_embeddings_bl7"; ENTRY="evaluate.py" ;;
    ar_modern_bl7_v2_cross)     N_CHUNKS=4;  GRES="gpu:1";      CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="ar_modern_bl7_v2_cross_fold"; EXTRA="--emb-dir $SCRIPT_DIR/sensor_embeddings_bl7_v2"; ENTRY="evaluate.py" ;;
    # State-balanced decoder retrain over the frozen BL-7 v1 encoder
    ar_modern_bl7_e0)           N_CHUNKS=4;  GRES="gpu:1";      CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="ar_modern_bl7_e0_fold"; EXTRA="--emb-dir $SCRIPT_DIR/sensor_embeddings_bl7"; ENTRY="evaluate.py" ;;
    # Factored opcode/operand decoder head over the frozen BL-7 v1 encoder
    ar_modern_bl7_e2factored)   N_CHUNKS=4;  GRES="gpu:1";      CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="ar_modern_bl7_e2factored_fold"; EXTRA="--emb-dir $SCRIPT_DIR/sensor_embeddings_bl7"; ENTRY="evaluate.py" ;;
    # Threshold-feature decoder conditioning over the frozen BL-7 v1 encoder
    ar_modern_bl7_e1thresh)     N_CHUNKS=4;  GRES="gpu:1";      CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="ar_modern_bl7_e1thresh_fold"; EXTRA="--emb-dir $SCRIPT_DIR/sensor_embeddings_bl7_thresh"; ENTRY="evaluate.py" ;;
    # Phase B (BL-7 sibling): GRPO post-step on the BL7+thresh CE ckpt. Same
    # emb dir as e1thresh; ckpt is grpo_best.pt; _sb suffix is the per-state-
    # bonus ablation.
    ar_modern_bl7_e1thresh_grpo|ar_modern_bl7_e1thresh_grpo_sb)
        N_CHUNKS=4;  GRES="gpu:1";      CKPT_NAME="grpo_best.pt"; FOLD_PREFIX="${VARIANT}_fold"
        EXTRA="--emb-dir $SCRIPT_DIR/sensor_embeddings_bl7_thresh"
        ENTRY="evaluate.py" ;;
    # Long-horizon (4000-tick) no-Perceiver encoder: joint encode-on-the-fly
    # eval (no --emb-dir). Decoder cross-attends to ~686 axial tokens.
    ar_modern_bl7_w4000_axial)  N_CHUNKS=4;  GRES="gpu:1";      CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="ar_modern_bl7_w4000_axial_fold"; EXTRA="--joint-encoder-ckpt $SCRIPT_DIR/models/bl7_w4000/bl7_best.pt --bl7-axial --bl7-window-size 4000 --batch-size 4"; ENTRY="evaluate.py" ;;
    # Same no-Perceiver eval but over the CONTRASTIVE (bl7_w4000_v2) encoder,
    # whose axial tokens retain long-horizon state ~4-5x better than data2vec.
    ar_modern_bl7_v2_w4000_axial)  N_CHUNKS=4;  GRES="gpu:1";   CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="ar_modern_bl7_v2_w4000_axial_fold"; EXTRA="--joint-encoder-ckpt $SCRIPT_DIR/models/bl7_w4000_v2/bl7_best.pt --bl7-axial --bl7-window-size 4000 --batch-size 4"; ENTRY="evaluate.py" ;;
    # 4000-tick ablation Step 1: prefix-K4 encoder-checkpoint probe. Pooled
    # (n,4,256) embeddings precomputed per encoder step; mean-pooled to 256 by
    # data.py at load. Per-fold emb dir overridden in the fold loop below.
    ar_modern_bl7_w4000_prefix_s*)
        N_CHUNKS=4;  GRES="gpu:1";      CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="${VARIANT}_fold"
        EXTRA="--emb-dir $SCRIPT_DIR/sensor_embeddings_${VARIANT}_fold_0"
        ENTRY="evaluate.py" ;;
    # 4000-tick ablation Step 2: cross-attn K-sweep on E*=step-1000. Joint-trained
    # K-latent pool; (n,K,256) embeddings precomputed per fold (overridden below).
    ar_modern_bl7_w4000_k4|ar_modern_bl7_w4000_k8|ar_modern_bl7_w4000_k16|ar_modern_bl7_w4000_k32|ar_modern_bl7_w4000_k64)
        N_CHUNKS=4;  GRES="gpu:1";      CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="${VARIANT}_fold"
        EXTRA="--emb-dir $SCRIPT_DIR/sensor_embeddings_${VARIANT}_fold_0"
        ENTRY="evaluate.py" ;;
    # Same cross-attn K-pool eval but over the CONTRASTIVE (bl7_w4000_v2) encoder.
    ar_modern_bl7_v2_w4000_k4|ar_modern_bl7_v2_w4000_k8|ar_modern_bl7_v2_w4000_k16|ar_modern_bl7_v2_w4000_k32|ar_modern_bl7_v2_w4000_k64)
        N_CHUNKS=8;  GRES="gpu:1";      CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="${VARIANT}_fold"
        EXTRA="--emb-dir $SCRIPT_DIR/sensor_embeddings_${VARIANT}_fold_0"
        ENTRY="evaluate.py" ;;
    # Same cross-attn K-pool eval but over the HuBERT (bl7_w4000_hubert) encoder.
    ar_modern_bl7_hub_w4000_k4|ar_modern_bl7_hub_w4000_k8|ar_modern_bl7_hub_w4000_k16|ar_modern_bl7_hub_w4000_k32|ar_modern_bl7_hub_w4000_k64)
        N_CHUNKS=8;  GRES="gpu:1";      CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="${VARIANT}_fold"
        EXTRA="--emb-dir $SCRIPT_DIR/sensor_embeddings_${VARIANT}_fold_0"
        ENTRY="evaluate.py" ;;
    # Phase A: BL-2 (Chronos-2 finetuned per W) at different window sizes.
    # One emb dir per W (sensor_embeddings_w{W}/), shared across folds.
    # Phase B suffixes:
    #   _thresh         -> CE-trained AR-modern over thresh-augmented emb (272-d)
    #   _thresh_grpo    -> GRPO-tuned on top (ckpt = grpo_best.pt, same emb dir)
    #   _thresh_grpo_sb -> GRPO with --use-state-bonus ablation
    ar_modern_bl2_w*)
        N_CHUNKS=4;  GRES="gpu:1";      FOLD_PREFIX="${VARIANT}_fold"
        W_SUFFIX="${VARIANT#ar_modern_bl2_}"
        if [[ "$W_SUFFIX" == *_grpo_sb ]]; then
            CKPT_NAME="grpo_best.pt"; EMB_SUFFIX="${W_SUFFIX%_grpo_sb}"
        elif [[ "$W_SUFFIX" == *_grpo_v2 ]]; then
            CKPT_NAME="grpo_best.pt"; EMB_SUFFIX="${W_SUFFIX%_grpo_v2}"
        elif [[ "$W_SUFFIX" == *_grpo ]]; then
            CKPT_NAME="grpo_best.pt"; EMB_SUFFIX="${W_SUFFIX%_grpo}"
        elif [[ "$W_SUFFIX" == *_transfer ]]; then
            # transfer variant: different train split, same thresh embeddings
            CKPT_NAME="ar_modern_best.pt"; EMB_SUFFIX="${W_SUFFIX%_transfer}"
        else
            CKPT_NAME="ar_modern_best.pt"; EMB_SUFFIX="$W_SUFFIX"
        fi
        EXTRA="--emb-dir $SCRIPT_DIR/sensor_embeddings_${EMB_SUFFIX}"
        ENTRY="evaluate.py" ;;
    # E2 K-sweep variants (joint-trained; emb dir derived from VARIANT name or EMB_DIR env override)
    ar_modern_bl7_k4|ar_modern_bl7_k8|ar_modern_bl7_k16|ar_modern_bl7_k32|ar_modern_bl7_k64|ar_modern_bl7_axial)
        N_CHUNKS=4;  GRES="gpu:1";      CKPT_NAME="ar_modern_best.pt"; FOLD_PREFIX="${VARIANT}_fold"
        DEFAULT_EMB_DIR="$SCRIPT_DIR/sensor_embeddings_${VARIANT}"
        EXTRA="--emb-dir ${EMB_DIR:-$DEFAULT_EMB_DIR}"
        ENTRY="evaluate.py" ;;
    bl3|bl4|bl6)                N_CHUNKS=4;  GRES="gpu:1";      CKPT_NAME="${VARIANT}_best.pt"; FOLD_PREFIX="${VARIANT}_fold"; EXTRA=""; ENTRY="evaluate.py" ;;
    bl3b)                       N_CHUNKS=8;  GRES="gpu:1";      CKPT_NAME="bl3b_best.pt"; FOLD_PREFIX="bl3b_fold"; EXTRA=""; ENTRY="bl3b_eval.py" ;;
    *) echo "ERROR: variant must be mdlm | ar_modern | ar_modern_bl7{,_v2,_cross,_v2_cross,_k{4,8,16,32,64},_axial,_w4000_axial,_w4000_prefix_s<N>} | bl3 | bl4 | bl6 | bl3b" >&2; exit 1 ;;
esac

JOBS=""
for fold_idx in "${FOLDS[@]}"; do
    fold_dir="$SCRIPT_DIR/models/${FOLD_PREFIX}_${fold_idx}"
    case "$VARIANT" in
        mdlm)      result_dir="$SCRIPT_DIR/results/mdlm_strat100k_fold_${fold_idx}" ;;
        ar_modern) result_dir="$SCRIPT_DIR/results/ar_modern_strat100k_fold_${fold_idx}" ;;
        *)         result_dir="$SCRIPT_DIR/results/${VARIANT}_strat100k_fold_${fold_idx}" ;;
    esac
    # E2 K-sweep / joint-trained variants need per-fold emb-dir (each fold's
    # trained pool produces different embeddings). Override EXTRA's --emb-dir
    # inside this fold loop for those variants.
    case "$VARIANT" in
        ar_modern_bl7_k4|ar_modern_bl7_k8|ar_modern_bl7_k16|ar_modern_bl7_k32|ar_modern_bl7_k64|ar_modern_bl7_axial|ar_modern_bl7_w4000_prefix_s*|ar_modern_bl7_w4000_k4|ar_modern_bl7_w4000_k8|ar_modern_bl7_w4000_k16|ar_modern_bl7_w4000_k32|ar_modern_bl7_w4000_k64|ar_modern_bl7_v2_w4000_k4|ar_modern_bl7_v2_w4000_k8|ar_modern_bl7_v2_w4000_k16|ar_modern_bl7_v2_w4000_k32|ar_modern_bl7_v2_w4000_k64|ar_modern_bl7_hub_w4000_k4|ar_modern_bl7_hub_w4000_k8|ar_modern_bl7_hub_w4000_k16|ar_modern_bl7_hub_w4000_k32|ar_modern_bl7_hub_w4000_k64)
            PER_FOLD_EMB="$SCRIPT_DIR/sensor_embeddings_${VARIANT}_fold_${fold_idx}"
            EXTRA="--emb-dir ${PER_FOLD_EMB}"
            ;;
    esac
    mkdir -p "$result_dir"
    chunk_jobs=""
    for chunk_id in $(seq 0 $((N_CHUNKS - 1))); do
        # Chunk slices the post-stratification 100K ordered list.
        # We don't know the exact chunk boundary in advance because state-0/2
        # counts vary slightly per fold; using TARGET/N_CHUNKS as an upper
        # bound is fine (the loader clips at len(stratified subset)).
        chunk_size=$((TARGET / N_CHUNKS))
        start=$((chunk_id * chunk_size))
        end=$((start + chunk_size))
        JOB=$(sbatch --parsable \
            --job-name="${VARIANT}_strat_f${fold_idx}_c${chunk_id}" \
            --partition=short --gres="$GRES" --cpus-per-task=4 --mem=64G --time="$CHUNK_TIME" \
            --exclude=gpu-6-[01-20] \
            --output="$result_dir/slurm_${VARIANT}_strat_c${chunk_id}_%j.log" \
            --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
CKPT=$fold_dir/$CKPT_NAME
echo \"Stratified chunk $chunk_id [\$start, \$end) of ~${TARGET}\"
python3 -u $ENTRY \
    --checkpoint \$CKPT \
    --split test \
    --sessions ${TEST_SESSIONS[*]} \
    --device cuda \
    --stratified-target $TARGET \
    --n-samples-start $start \
    --n-samples-end $end \
    --num-workers 0 \
    --save-raw $result_dir/${VARIANT}_eval_test_chunk_${chunk_id}.npz \
    --result-dir $result_dir $EXTRA
")
        chunk_jobs="${chunk_jobs}:${JOB}"
        echo "  $VARIANT strat f${fold_idx} c${chunk_id} -> $JOB"
        JOBS="$JOBS $JOB"
    done
    MERGE=$(sbatch --parsable \
        --job-name="${VARIANT}_strat_f${fold_idx}_merge" \
        --partition=short --cpus-per-task=2 --mem=32G --time=01:00:00 \
        --dependency="afterok${chunk_jobs}" \
        --output="$result_dir/slurm_${VARIANT}_strat_merge_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export TOKENIZERS_PARALLELISM=false
python3 -u merge_chunks.py --chunks $result_dir/${VARIANT}_eval_test_chunk_*.npz --result-dir $result_dir --name $VARIANT --split test
")
    echo "  $VARIANT strat f${fold_idx} merge -> $MERGE"
    JOBS="$JOBS $MERGE"
done
echo ""
echo "Submitted $VARIANT stratified 100K:$JOBS"
