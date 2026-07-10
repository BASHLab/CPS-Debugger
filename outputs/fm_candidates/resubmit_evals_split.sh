#!/bin/bash
# Submit eval jobs split test/val per fold (2 jobs per fold), on any-GPU for
# AR variants (timing not strictly comparable; we'll do a separate H100
# timing pass later). For baselines we keep H100 since that's where the
# previous numbers were measured.

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")

submit_split() {
    local variant=$1            # bl3 / bl4 / bl6 / ar_modern
    local fold_idx=$2
    local split=$3               # test or val
    local gres=$4                # "gpu:1" or "gpu:H100:1"

    if [ "$variant" = "ar_modern" ]; then
        local fold_dir="$SCRIPT_DIR/models/ar_modern_fold_${fold_idx}"
        local result_dir="$SCRIPT_DIR/results/ar_modern_fold_${fold_idx}"
        local ckpt_glob="ar_modern"
    else
        local fold_dir="$SCRIPT_DIR/models/${variant}_fold_${fold_idx}"
        local result_dir="$SCRIPT_DIR/results/${variant}_fold_${fold_idx}"
        local ckpt_glob="${variant}"
    fi
    mkdir -p "$result_dir"

    if [ "$split" = "test" ]; then
        local sessions_list="${TEST_SESSIONS[*]}"
    else
        local sessions_list="${VAL_SESSIONS[*]}"
    fi

    sbatch --parsable \
        --job-name="${variant}_f${fold_idx}_${split}" \
        --partition=short \
        --gres="$gres" \
        --cpus-per-task=4 \
        --mem=64G \
        --time=06:00:00 \
        --exclude=gpu-6-[01-20] \
        --output="$result_dir/slurm_${variant}_${split}_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

CKPT=$fold_dir/${ckpt_glob}_best.pt
if [ ! -f \$CKPT ]; then
    CKPT=\$(ls -t $fold_dir/${ckpt_glob}_final_*.pt 2>/dev/null | head -1)
fi
echo \"Using checkpoint: \$CKPT\"

echo '=== Fold $fold_idx ($variant): $split ==='
python3 -u evaluate.py --checkpoint \$CKPT --split $split --sessions $sessions_list --device cuda --n-samples 30000 --num-workers 0 --result-dir $result_dir
"
}

submit_baseline_split() {
    local fold_idx=$1
    local baseline=$2          # unigram / copy_modal / copy_previous
    local split=$3
    local result_dir="$SCRIPT_DIR/results/baselines_fold_${fold_idx}"
    mkdir -p "$result_dir"

    # Train sessions per fold (need for baseline-train-sessions arg)
    local fold_train_var="FOLD_TRAIN_${fold_idx}[@]"
    local train_sess=("${!fold_train_var}")

    if [ "$split" = "test" ]; then
        local sessions_list="${TEST_SESSIONS[*]}"
    else
        local sessions_list="${VAL_SESSIONS[*]}"
    fi

    sbatch --parsable \
        --job-name="base_${baseline}_f${fold_idx}_${split}" \
        --partition=short \
        --gres=gpu:H100:1 \
        --cpus-per-task=4 \
        --mem=32G \
        --time=04:00:00 \
        --exclude=gpu-6-[01-20] \
        --output="$result_dir/slurm_${baseline}_${split}_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

echo '=== Fold $fold_idx ($baseline): $split ==='
python3 -u evaluate.py --baseline $baseline --split $split --sessions $sessions_list --baseline-train-sessions ${train_sess[*]} --device cuda --n-samples 30000 --num-workers 0 --result-dir $result_dir
"
}

# Train-session lists (same as submit_baselines_5fold.sh)
FOLD_TRAIN_0=("2025-03-17_10-36-44" "2025-03-20_09-31-56")
FOLD_TRAIN_1=("2025-03-17_10-51-58" "2025-03-20_09-46-30")
FOLD_TRAIN_2=("2025-03-17_11-06-39" "2025-03-20_10-03-00")
FOLD_TRAIN_3=("2025-03-18_12-39-10" "2025-03-19_11-10-13")
FOLD_TRAIN_4=("2025-03-19_10-05-47" "2025-03-19_10-37-56")

JOBS=""

# AR-modern (5 folds × 2 splits = 10 jobs, any-GPU)
for fold_idx in 0 1 2 3 4; do
    for split in test val; do
        JOB=$(submit_split ar_modern $fold_idx $split "gpu:1")
        echo "  ar_modern f$fold_idx $split -> $JOB"; JOBS="$JOBS $JOB"
    done
done

# BL-6 (5 folds × 2 splits = 10 jobs, any-GPU)
for fold_idx in 0 1 2 3 4; do
    for split in test val; do
        JOB=$(submit_split bl6 $fold_idx $split "gpu:1")
        echo "  bl6 f$fold_idx $split -> $JOB"; JOBS="$JOBS $JOB"
    done
done

# BL-4 fold 3, 4 only (folds 0/1/2 already running combined)
for fold_idx in 3 4; do
    for split in test val; do
        JOB=$(submit_split bl4 $fold_idx $split "gpu:1")
        echo "  bl4 f$fold_idx $split -> $JOB"; JOBS="$JOBS $JOB"
    done
done

# Baselines (5 folds × 3 baselines × 2 splits = 30 jobs, H100-pinned)
for fold_idx in 0 1 2 3 4; do
    for baseline in unigram copy_modal copy_previous; do
        for split in test val; do
            JOB=$(submit_baseline_split $fold_idx $baseline $split)
            echo "  base $baseline f$fold_idx $split -> $JOB"; JOBS="$JOBS $JOB"
        done
    done
done

echo ""
echo "Total resubmitted:$JOBS"
