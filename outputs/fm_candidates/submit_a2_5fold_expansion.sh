#!/bin/bash
# A2 (IAAI paper plan) — 5-fold expansion of BL2-4s and BL2-4s+thresh.
#
# Fold-0 already done (BL2-4s 0.0246, BL2-4s+thresh 0.0197). This script runs
# folds 1-4 for both variants so the paper headline is 5-fold like every
# other featured leaderboard row.
#
# Chain per the canonical 5-fold protocol (submit_ar_modern_5fold.sh):
#   1. Precompute W=4000 Chronos embeddings for the 8 missing fold-1..4
#      train sessions (slimllm_bw env; ~2.5 h/session on H100-class).
#   2. Thresh-augment those sessions (CPU, quick).
#   3. Per fold f in 1..4: AR-modern train plain (512-d) + thresh (528-d).
#   4. Stratified-100K eval triggers per (variant, fold).

set -euo pipefail
SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

PARTITION="${PARTITION:-short}"
SEED="${SEED:-42}"
W=4000
EMB="$SCRIPT_DIR/sensor_embeddings_w${W}"
EMB_T="$SCRIPT_DIR/sensor_embeddings_w${W}_thresh"
CHRONOS_CKPT="$SCRIPT_DIR/models/chronos2_finetuned_w${W}"

VAL_SESSIONS=("2025-03-21_11-21-42" "2025-03-25_13-23-42")
TEST_SESSIONS=("2025-03-25_13-39-06" "2025-03-26_11-03-04")
FOLD_TRAIN_1=("2025-03-17_10-51-58" "2025-03-20_09-46-30")
FOLD_TRAIN_2=("2025-03-17_11-06-39" "2025-03-20_10-03-00")
FOLD_TRAIN_3=("2025-03-18_12-39-10" "2025-03-19_11-10-13")
FOLD_TRAIN_4=("2025-03-19_10-05-47" "2025-03-19_10-37-56")

MISSING_SESS=("${FOLD_TRAIN_1[@]}" "${FOLD_TRAIN_2[@]}" "${FOLD_TRAIN_3[@]}" "${FOLD_TRAIN_4[@]}")

# ── Stage 1: per-session precompute (parallel; skip if emb exists) ────────
PP_JOBS=""
for sess in "${MISSING_SESS[@]}"; do
    if [ -f "$EMB/${sess}_emb.npy" ]; then
        echo "precompute $sess: already present, skip"
        continue
    fi
    J=$(sbatch --parsable \
        --partition="$PARTITION" --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=16:00:00 \
        --exclude=gpu-6-[01-20] \
        --job-name="a2_pp_${sess}" \
        --output="${EMB}/slurm_precompute_${sess}_%j.log" \
        --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm_bw/bin:\$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -u sensor_encoder.py --sessions $sess \
    --window-size $W --chronos-ckpt $CHRONOS_CKPT --output-dir $EMB \
    --batch-size 64 --device cuda")
    echo "precompute $sess -> $J"
    PP_JOBS="${PP_JOBS}:${J}"
done

# ── Stage 2: thresh-augment the 8 sessions (afterok all precompute) ───────
DEP=""
[ -n "$PP_JOBS" ] && DEP="--dependency=afterok${PP_JOBS}"
TH_JOB=$(sbatch --parsable $DEP \
    --partition=quick --cpus-per-task=4 --mem=32G --time=01:00:00 \
    --job-name="a2_thresh_pp" \
    --output="${EMB_T}/slurm_thresh_pp_a2_%j.log" \
    --wrap="
cd $SCRIPT_DIR
export PATH=/home/simran/.conda/envs/slimllm/bin:\$PATH
python3 -u precompute_thresh_features.py \
    --sessions ${MISSING_SESS[*]} \
    --src-emb-dir $EMB --dst-emb-dir $EMB_T")
echo "thresh-augment -> $TH_JOB"

# ── Stages 3+4: per fold, both variants ───────────────────────────────────
for fold_idx in 1 2 3 4; do
    var="FOLD_TRAIN_${fold_idx}[@]"
    TRAIN_SESS=("${!var}")

    for variant in plain thresh; do
        if [ "$variant" = "plain" ]; then
            VNAME="ar_modern_bl2_w${W}"
            VEMB="$EMB"; DSE=512
        else
            VNAME="ar_modern_bl2_w${W}_thresh"
            VEMB="$EMB_T"; DSE=528
        fi
        FOLD_DIR="$SCRIPT_DIR/models/${VNAME}_fold_${fold_idx}"
        mkdir -p "$FOLD_DIR"

        TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$TH_JOB \
            --partition="$PARTITION" \
            --job-name="${VNAME}_f${fold_idx}_train" \
            --output="${FOLD_DIR}/slurm_train_%j.log" \
            run_ar_modern_train.sh \
            --seed "$SEED" \
            --train-sessions "${TRAIN_SESS[@]}" \
            --val-sessions "${VAL_SESSIONS[@]}" \
            --emb-dir "$VEMB" \
            --d-sensor-emb "$DSE" \
            --model-dir "$FOLD_DIR" \
            --max-steps 4000 --warmup-steps 500 --eval-every 500 --save-every 1000)
        echo "fold $fold_idx $variant train -> $TRAIN_JOB"

        EVAL_TRIG=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB \
            --partition="$PARTITION" --cpus-per-task=2 --mem=4G --time=00:15:00 \
            --output="${FOLD_DIR}/strat_trigger_%j.log" \
            --wrap="cd $SCRIPT_DIR && bash submit_stratified_100k.sh $VNAME $fold_idx")
        echo "fold $fold_idx $variant eval trig -> $EVAL_TRIG"
    done
done

echo ""
echo "A2 5-fold expansion submitted."
