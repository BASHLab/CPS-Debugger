#!/bin/bash
# One-time setup: creates grpo conda env with torch 2.4 + verl
# Run this ONCE before submitting the GRPO SLURM job.
set -e

ENV_NAME=grpo
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
ENV_PATH="$CONDA_BASE/envs/$ENV_NAME"

if [[ -d "$ENV_PATH" ]]; then
    echo "grpo env already exists at $ENV_PATH"
    exit 0
fi

echo "Creating conda env: $ENV_NAME (Python 3.10)..."
conda create -n "$ENV_NAME" python=3.10 -y

echo "Installing PyTorch 2.4 (CUDA 12.1)..."
"$ENV_PATH/bin/pip" install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 --quiet

echo "Installing verl + dependencies..."
"$ENV_PATH/bin/pip" install verl transformers accelerate datasets \
    tensordict peft bitsandbytes flash-attn --quiet || \
"$ENV_PATH/bin/pip" install verl transformers accelerate datasets \
    tensordict peft bitsandbytes --quiet

echo "Verifying..."
"$ENV_PATH/bin/python" -c "
import torch, verl, tensordict
print(f'torch: {torch.__version__}')
print(f'tensordict: {tensordict.__version__}')
print(f'verl: {getattr(verl, \"__version__\", \"ok\")}')
print('CUDA:', torch.cuda.is_available())
"

echo "Done. Use: /home/simran/.conda/envs/grpo/bin/python3"
