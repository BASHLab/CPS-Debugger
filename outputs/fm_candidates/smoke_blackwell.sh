#!/bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --nodelist=gpu-6-01,gpu-6-02,gpu-6-06,gpu-6-07,gpu-6-12,gpu-6-13,gpu-6-15,gpu-6-17,gpu-6-18
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates/smoke_blackwell_%j.log

# Smoke test: verify torch nightly cu128 in slimllm_bw works on RTX PRO 6000 B (sm_120).

export PATH="/home/simran/.conda/envs/slimllm_bw/bin:$PATH"

echo "=== Node ==="
hostname
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

echo ""
echo "=== Torch info ==="
python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('archs:', torch.cuda.get_arch_list())
print('device count:', torch.cuda.device_count())
print('device name:', torch.cuda.get_device_name(0))
print('compute cap:', torch.cuda.get_device_capability(0))
"

echo ""
echo "=== matmul test ==="
python -c "
import torch
x = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
y = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
z = x @ y
torch.cuda.synchronize()
print('matmul shape:', z.shape, 'dtype:', z.dtype, 'norm:', z.float().norm().item())
"

echo ""
echo "=== SDPA test (transformer kernel) ==="
python -c "
import torch
import torch.nn.functional as F
B, H, L, D = 2, 8, 256, 64
q = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
torch.cuda.synchronize()
print('SDPA out:', out.shape, 'norm:', out.float().norm().item())
"

echo ""
echo "Exit: $?"
