"""Deterministic CUDA inference-time measurement.

Replaces `time.time()` wrappers that measure the async kernel-launch queue,
not GPU work. For a shared SLURM cluster without clock-locking permissions,
our best tools are: CUDA Events (GPU-side timestamps), mandatory warmup,
GC suppression during timing, fixed cudnn flags, and robust aggregation.
"""
from __future__ import annotations

import gc
import os
import statistics
import subprocess
from typing import Any, Callable, Dict, List, Tuple

import torch


def set_stable_timing_mode() -> None:
    """Fix kernel-selection state so numbers are reproducible across runs.

    cudnn.benchmark=False avoids autotuner picking different algos on different
    runs depending on workspace availability. TF32 stays on (bf16 autocast
    dominates anyway; TF32 only touches the rare fp32 fallback paths).
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False  # still allow fast kernels
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def warmup(fn: Callable, n: int = 3) -> None:
    """Run fn n times and discard results. Triggers lazy init + cuDNN cache."""
    for _ in range(n):
        _ = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class gc_paused:
    """Disable Python GC for the duration of a with-block."""

    def __enter__(self):
        self._was_enabled = gc.isenabled()
        if self._was_enabled:
            gc.collect()
            gc.disable()
        return self

    def __exit__(self, *_):
        if self._was_enabled:
            gc.enable()


def time_cuda(fn: Callable) -> Tuple[Any, float]:
    """One CUDA-synchronized call. Returns (result, elapsed_ms).

    Use AFTER warmup(). Wrap the enclosing loop with `with gc_paused():`.
    """
    if not torch.cuda.is_available():
        import time as _t
        t0 = _t.perf_counter()
        out = fn()
        return out, (_t.perf_counter() - t0) * 1000.0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end)


def summarize_ms(ms: List[float]) -> Dict[str, Any]:
    """Robust summary: median + quantiles + min/max. Mean is for completeness."""
    if not ms:
        return {"n": 0}
    xs = sorted(ms)
    n = len(xs)

    def pct(p: float) -> float:
        k = (p / 100.0) * (n - 1)
        lo = int(k)
        hi = min(lo + 1, n - 1)
        frac = k - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac

    return {
        "n": n,
        "median_ms": statistics.median(xs),
        "mean_ms": statistics.fmean(xs),
        "std_ms": statistics.pstdev(xs) if n > 1 else 0.0,
        "min_ms": xs[0],
        "max_ms": xs[-1],
        "p50_ms": pct(50),
        "p90_ms": pct(90),
        "p99_ms": pct(99),
    }


def log_gpu_state(tag: str = "") -> Dict[str, Any]:
    """Snapshot GPU temp/clock/power for post-hoc filtering of noisy runs.

    Returns a dict with the reading; also prints it. Best called before warmup
    and after the timing loop so we can check whether thermal state drifted.
    """
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,clocks.sm,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5)
        fields = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        d = {
            "name": fields[0],
            "temp_c": float(fields[1]),
            "sm_mhz": float(fields[2]),
            "power_w": float(fields[3]),
            "power_limit_w": float(fields[4]),
            "host": os.environ.get("HOSTNAME") or os.environ.get("SLURMD_NODENAME", ""),
            "job_id": os.environ.get("SLURM_JOB_ID", ""),
        }
        prefix = f"[gpu:{tag}]" if tag else "[gpu]"
        print(f"  {prefix} {d['name']}  {d['temp_c']:.0f}C  "
              f"{d['sm_mhz']:.0f}MHz  {d['power_w']:.0f}W / {d['power_limit_w']:.0f}W")
        return d
    except Exception as e:
        print(f"  [gpu:{tag}] could not read: {e}")
        return {}
