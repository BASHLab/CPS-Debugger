#!/bin/bash
# E5 submission: train AR-modern with state-token prefix + CFG support,
# 5 folds on BL-7 v1 sensor embeddings + precomputed per-tick state_pred.
#
# Prerequisites:
#   - sensor_embeddings_bl7/<sess>_emb.npy (precomputed BL-7 v1 embeddings)
#   - sensor_embeddings_bl7/<sess>_state_pred.npy (from bl7_probes.py
#       dump-state-predictions; covers all 15 sessions)
#   - models/bl7_state_probe.pt (the probe checkpoint)
#
# After all 5 train jobs succeed, the CFG-eval-sweep chain auto-fires
# (eval_cfg.sh per fold per γ value).

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

FOLDS="${FOLDS:-0 1 2 3 4}"
SEED="${SEED:-42}"
PARTITION="${PARTITION:-short}"
EMB_DIR="${EMB_DIR:-$SCRIPT_DIR/sensor_embeddings_bl7}"

VARIANT_NAME="ar_modern_bl7_statecfg"

# Sanity: state-pred files must exist for the val + test sessions
for sess in 2025-03-21_11-21-42 2025-03-25_13-23-42 2025-03-25_13-39-06 2025-03-26_11-03-04; do
    test -f "$EMB_DIR/${sess}_state_pred.npy" || {
        echo "ERROR: $EMB_DIR/${sess}_state_pred.npy missing. Run bl7_probes.py dump-state-predictions first." >&2
        exit 1
    }
done
echo "All val+test state_pred files present in $EMB_DIR."

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")
FOLD_TRAIN_0=("2025-03-17_10-36-44" "2025-03-20_09-31-56")
FOLD_TRAIN_1=("2025-03-17_10-51-58" "2025-03-20_09-46-30")
FOLD_TRAIN_2=("2025-03-17_11-06-39" "2025-03-20_10-03-00")
FOLD_TRAIN_3=("2025-03-18_12-39-10" "2025-03-19_11-10-13")
FOLD_TRAIN_4=("2025-03-19_10-05-47" "2025-03-19_10-37-56")

declare -A TRAIN_BY_FOLD
TRAIN_JOBS=""
for fold in $FOLDS; do
    var="FOLD_TRAIN_${fold}[@]"
    TRAIN_SESS=("${!var}")

    FOLD_DIR="$SCRIPT_DIR/models/${VARIANT_NAME}_fold_${fold}"
    RESULT_DIR="$SCRIPT_DIR/results/${VARIANT_NAME}_fold_${fold}"
    mkdir -p "$FOLD_DIR" "$RESULT_DIR"

    TRAIN_JOB=$(sbatch --parsable \
        --partition="$PARTITION" \
        --gres=gpu:1 \
        --time=04:00:00 \
        --exclude=gpu-6-[01-20] \
        --job-name="${VARIANT_NAME}_f${fold}_train" \
        --output="$FOLD_DIR/slurm_${VARIANT_NAME}_train_%j.log" \
        run_ar_modern_train.sh \
        --seed "$SEED" \
        --train-sessions "${TRAIN_SESS[@]}" \
        --val-sessions "${VAL_SESSIONS[@]}" \
        --emb-dir "$EMB_DIR" \
        --d-sensor-emb 256 --n-sensor-tokens 1 \
        --model-dir "$FOLD_DIR" \
        --max-steps 4000 --warmup-steps 500 --eval-every 500 --save-every 1000 \
        --use-state-token --cond-dropout-p 0.10)
    echo "  $VARIANT_NAME fold $fold train -> $TRAIN_JOB"
    TRAIN_JOBS="$TRAIN_JOBS $TRAIN_JOB"
    TRAIN_BY_FOLD[$fold]=$TRAIN_JOB

    # Per-fold CFG-sweep eval (4 γ values), each waiting for this fold's train
    for GAMMA in 1.0 1.5 2.0 3.0; do
        GAMMA_TAG=$(echo $GAMMA | sed 's/\./_/g')
        EVAL_JOB=$(sbatch --parsable \
            --dependency=afterok:${TRAIN_JOB} \
            --partition="$PARTITION" \
            --gres=gpu:1 \
            --cpus-per-task=4 --mem=64G --time=04:00:00 \
            --exclude=gpu-6-[01-20] \
            --job-name="${VARIANT_NAME}_f${fold}_eval_g${GAMMA_TAG}" \
            --output="$RESULT_DIR/slurm_eval_g${GAMMA_TAG}_%j.log" \
            --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CKPT=$FOLD_DIR/ar_modern_best.pt
python3 -u evaluate.py \
    --checkpoint \$CKPT \
    --split test \
    --sessions ${TEST_SESSIONS[*]} \
    --device cuda \
    --emb-dir $EMB_DIR \
    --n-samples 1000 \
    --guidance-scale $GAMMA \
    --result-dir $RESULT_DIR \
    --result-suffix _g${GAMMA_TAG}
")
        echo "  $VARIANT_NAME fold $fold eval γ=$GAMMA -> $EVAL_JOB"
    done
done

echo ""
echo "E5 state-token + CFG submitted. 5 folds × 4 γ values = 20 eval jobs after 5 train jobs."
echo "Monitor: squeue -u \$USER --format='%.10i %.30j %.8T %.10M %R'"
