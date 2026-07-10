#!/bin/bash
# Submit BL-3b 5-fold training+eval. Same fold layout as BL-3.
# DEPENDS on the 30fps frame caches (frames_30fps/{session}.npy) being
# produced first by submit_frames_all.sh.

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")

FOLD_TRAIN_0=("2025-03-17_10-36-44" "2025-03-20_09-31-56")
FOLD_TRAIN_1=("2025-03-17_10-51-58" "2025-03-20_09-46-30")
FOLD_TRAIN_2=("2025-03-17_11-06-39" "2025-03-20_10-03-00")
FOLD_TRAIN_3=("2025-03-18_12-39-10" "2025-03-19_11-10-13")
FOLD_TRAIN_4=("2025-03-19_10-05-47" "2025-03-19_10-37-56")

SEED="${SEED:-42}"
PARTITION="${PARTITION:-short}"
DEPENDENCY="${DEPENDENCY:-}"     # optional; e.g. afterok:1986711:1986712:...

echo "Submitting 5-fold BL-3b training (seed=$SEED, partition=$PARTITION)..."
TRAIN_JOBS=""

for fold_idx in 0 1 2 3 4; do
    var="FOLD_TRAIN_${fold_idx}[@]"
    TRAIN_SESS=("${!var}")

    FOLD_DIR="$SCRIPT_DIR/models/bl3b_fold_${fold_idx}"
    RESULT_DIR="$SCRIPT_DIR/results/bl3b_fold_${fold_idx}"
    mkdir -p "$FOLD_DIR" "$RESULT_DIR"

    DEP_ARG=""
    [ -n "$DEPENDENCY" ] && DEP_ARG="--dependency=$DEPENDENCY"

    TRAIN_JOB=$(sbatch --parsable \
        $DEP_ARG \
        --partition="$PARTITION" \
        --job-name="bl3b_f${fold_idx}_train" \
        --output="$FOLD_DIR/slurm_bl3b_train_%j.log" \
        run_bl3b_train.sh \
        --seed "$SEED" \
        --train-sessions "${TRAIN_SESS[@]}" \
        --val-sessions "${VAL_SESSIONS[@]}" \
        --model-dir "$FOLD_DIR" \
        --max-steps 2000 --warmup-steps 250 --eval-every 250 --save-every 500)
    echo "  Fold $fold_idx train: $TRAIN_JOB"

    EVAL_JOB=$(sbatch --parsable \
        --dependency=afterok:${TRAIN_JOB} \
        --job-name="bl3b_f${fold_idx}_eval" \
        --partition=short \
        --gres=gpu:H100:1 \
        --cpus-per-task=4 \
        --mem=80G \
        --time=06:00:00 \
        --exclude=gpu-6-[01-20] \
        --output="$RESULT_DIR/slurm_bl3b_eval_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CKPT=$FOLD_DIR/bl3b_best.pt
if [ ! -f \$CKPT ]; then
    CKPT=\$(ls -t $FOLD_DIR/bl3b_final_*.pt 2>/dev/null | head -1)
fi
echo \"Using checkpoint: \$CKPT\"
echo '=== Fold $fold_idx (BL-3b): test ==='
python3 -u evaluate.py --checkpoint \$CKPT --split test --sessions ${TEST_SESSIONS[*]} --device cuda --n-samples 30000 --result-dir $RESULT_DIR
echo '=== Fold $fold_idx (BL-3b): val ==='
python3 -u evaluate.py --checkpoint \$CKPT --split val --sessions ${VAL_SESSIONS[*]} --device cuda --n-samples 30000 --result-dir $RESULT_DIR
")
    echo "  Fold $fold_idx eval:  $EVAL_JOB"
    TRAIN_JOBS="$TRAIN_JOBS $TRAIN_JOB"
done

echo ""
echo "All BL-3b folds submitted. Train jobs:$TRAIN_JOBS"
