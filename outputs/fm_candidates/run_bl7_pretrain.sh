#!/bin/bash
#SBATCH --job-name=bl7_pretrain
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates/results/bl7_pretrain_%j.log
#
# IMPORTANT: ALWAYS pass an explicit GRES type on the sbatch CLI.
# Do NOT use `--gres=gpu:1` (any-GPU) — it can land on Blackwell (gpu-6-*),
# which has compute capability sm_120. The default `slimllm` env is built
# against PyTorch cu121 with archs up to sm_90 only and will fail with
# "no kernel image is available for execution on the device". For Blackwell,
# pass `--gres=gpu:rtx_pro_6000_b:1` AND `run_bl7_pretrain.sh --use-blackwell-env`.
#
# Recommended invocations:
#   sbatch --gres=gpu:H200:1 run_bl7_pretrain.sh                    # full pretrain target
#   sbatch --gres=gpu:H100:1 run_bl7_pretrain.sh                    # if H100 frees up
#   sbatch --gres=gpu:L40S:1 run_bl7_pretrain.sh                    # ample availability, slower
#   sbatch --gres=gpu:H200:1 --time=00:30:00 \
#          run_bl7_pretrain.sh --sanity-check                       # sanity (200 steps)
#
# Blackwell variant (PyTorch nightly cu128 with sm_120):
#   sbatch --gres=gpu:rtx_pro_6000_b:1 run_bl7_pretrain.sh --use-blackwell-env

SCRIPT_DIR="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates"
cd "$SCRIPT_DIR"

# Default to the standard slimllm env; flip with --use-blackwell-env arg
USE_BLACKWELL=0
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "$arg" == "--use-blackwell-env" ]; then
        USE_BLACKWELL=1
    else
        EXTRA_ARGS+=("$arg")
    fi
done

if [ "$USE_BLACKWELL" == "1" ]; then
    export PATH="/home/simran/.conda/envs/slimllm_bw/bin:$PATH"
    echo "Using slimllm_bw env (PyTorch nightly cu128 for sm_120 Blackwell)"
else
    export PATH="/home/simran/.conda/envs/slimllm/bin:$PATH"
    echo "Using slimllm env (PyTorch stable)"
fi

# Determinism (per CPS-Debugger/CLAUDE.md)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

echo "=== Node ==="
hostname
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader

echo ""
echo "=== Torch info ==="
python3 -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('archs:', torch.cuda.get_arch_list())
print('device:', torch.cuda.get_device_name(0))
print('compute cap:', torch.cuda.get_device_capability(0))
"

echo ""
echo "=== Starting BL-7 pretrain ==="
python3 -u bl7_pretrain.py --device cuda --num-workers 4 "${EXTRA_ARGS[@]}"
EXIT=$?

echo ""
echo "=== Exit: $EXIT ==="
exit $EXIT
