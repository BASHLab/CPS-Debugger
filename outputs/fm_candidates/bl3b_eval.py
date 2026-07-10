"""Standalone eval for BL-3b (LoRA-adapted V-JEPA + AR-modern).

Mirrors evaluate.py's run_model_eval but:
  * loads BL3bModel from a bl3b_best.pt checkpoint (ar_state + lora_state)
  * uses BL3bDataset which returns sensor_emb + raw 64-frame clip per tick
  * compute_loss / sample take both sensor_emb and clip — V-JEPA forward
    happens inside the model at every sample call

Supports the same chunked + save-raw + merge-chunks pattern as evaluate.py
so the existing merge_chunks.py can pool chunks into a final JSON.
"""
import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make MAUVE/HF tokenizer fork-safe (vestigial; we dropped MAUVE but harmless)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import (
    ModelConfig, EvalConfig,
    VAL_SESSIONS, TEST_SESSIONS, EMB_DIR, DATA_DIR, RESULT_DIR,
)
from data import TraceDataset, load_token_mapping
from determinism import DEFAULT_SEED, set_deterministic
from metrics import compute_generation_metrics
from timing import gc_paused, log_gpu_state, summarize_ms, time_cuda, warmup
from bl3b_data import BL3bDataset
from bl3b_model import BL3bModel

# Same fixed seed used by evaluate.py for the deterministic permutation
EVAL_SUBSET_SEED = 1234567


def _resolve_sessions(args) -> List[str]:
    if args.sessions:
        return args.sessions
    if args.split == "val":
        return VAL_SESSIONS
    if args.split == "test":
        return TEST_SESSIONS
    raise ValueError(f"Unknown split: {args.split}")


def _load_split_dataset(args, mcfg: ModelConfig, sessions: List[str]):
    """Full TraceDataset → optional [start:end) slice of EVAL_SUBSET_SEED
    permutation OR stratified subset → wrap in BL3bDataset.

    Stratified mode (--stratified-target N): take ALL state-0 + ALL state-2 +
    (N − that) from state-1 for paper-quality minority-state CIs. Same logic
    as evaluate.py."""
    base = TraceDataset(
        sessions, mcfg.max_seq_len, mcfg.vocab_size,
        emb_dir=Path(args.emb_dir), data_dir=Path(args.data_dir),
        limit=None,
    )
    if args.stratified_target is not None and args.stratified_target > 0:
        fsm_arr = np.array(base.fsm_states, dtype=np.int64)
        idx_0 = np.where(fsm_arr == 0)[0]
        idx_2 = np.where(fsm_arr == 2)[0]
        idx_1_pool = np.where(fsm_arr == 1)[0]
        n_state1 = max(0, args.stratified_target - len(idx_0) - len(idx_2))
        rng = np.random.default_rng(EVAL_SUBSET_SEED)
        idx_1 = rng.permutation(idx_1_pool)[:n_state1]
        full_idx = np.sort(np.concatenate([idx_0, idx_1, idx_2]))
        if args.n_samples_end is not None and args.n_samples_end > 0:
            end = min(args.n_samples_end, len(full_idx))
        else:
            end = len(full_idx)
        start = max(0, args.n_samples_start)
        if start >= end:
            raise ValueError(f"empty stratified chunk: start={start} end={end}")
        idx = np.sort(full_idx[start:end])
        base = torch.utils.data.Subset(base, idx.tolist())
    elif args.n_samples is not None and args.n_samples > 0:
        rng = np.random.default_rng(EVAL_SUBSET_SEED)
        perm = rng.permutation(len(base))
        end = min(args.n_samples_end if args.n_samples_end and args.n_samples_end > 0
                  else args.n_samples, len(perm), args.n_samples)
        start = max(0, args.n_samples_start)
        if start >= end:
            raise ValueError(f"empty chunk: start={start} end={end}")
        idx = np.sort(perm[start:end])
        base = torch.utils.data.Subset(base, idx.tolist())
    return BL3bDataset(base)


def _collect_references(dl: DataLoader, pad_id: int):
    refs, lens, embs, fsm = [], [], [], []
    for batch in dl:
        x0 = batch["trace"]
        trace_len = batch["trace_len"]
        emb = batch["sensor_emb"]
        fs  = batch["fsm_state"]
        for i in range(x0.shape[0]):
            tl = int(trace_len[i])
            refs.append(x0[i, :tl].cpu().tolist())
            lens.append(tl)
            embs.append(emb[i].cpu().numpy())
            fsm.append(int(fs[i]))
    return refs, lens, (np.stack(embs) if embs else np.zeros((0, 0))), fsm


@torch.no_grad()
def _compute_tf_stats(model: BL3bModel, dl: DataLoader, device, pad_id: int) -> Dict[str, float]:
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    for batch in dl:
        x0 = batch["trace"].to(device)
        sensor_emb = batch["sensor_emb"].to(device)
        clip = batch["clip"].to(device)
        pad_mask = (x0 != pad_id).float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, m = model.compute_loss(x0, sensor_emb, clip, pad_mask)
        loss_sum += m["loss"]
        acc_sum  += m.get("acc_teacher_forced", 0.0)
        n += 1
    mean_loss = loss_sum / max(n, 1)
    return {
        "tf_loss":     mean_loss,
        "tf_token_acc": acc_sum / max(n, 1),
        "perplexity":   math.exp(min(mean_loss, 20)),
    }


def _time_generation(model: BL3bModel, dl: DataLoader, device, seq_len: int,
                     mode: str, warmup_calls: int = 3
                    ) -> Tuple[List[List[int]], List[float]]:
    greedy = (mode == "greedy")
    generated: List[List[int]] = []
    ms_list: List[float] = []
    # Warm-up on the first batch
    first = None
    for batch in dl:
        first = batch
        break
    if first is not None:
        sensor = first["sensor_emb"].to(device)
        clip   = first["clip"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            warmup(lambda: model.sample(sensor, clip, seq_len, greedy=greedy),
                   n=warmup_calls)

    with gc_paused():
        for batch in dl:
            x0 = batch["trace"]
            sensor = batch["sensor_emb"].to(device)
            clip   = batch["clip"].to(device)
            trace_len = batch["trace_len"]
            def _run():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    return model.sample(sensor, clip, seq_len, greedy=greedy)
            out, elapsed_ms = time_cuda(_run)
            ms_list.append(elapsed_ms)
            for i in range(x0.shape[0]):
                tl = int(trace_len[i])
                generated.append(out[i, :tl].cpu().tolist())
    return generated, ms_list


def _load_bl3b_from_ckpt(ckpt: dict, mcfg: ModelConfig, device) -> BL3bModel:
    cond_dim = ckpt["conditioning_dim"]
    if cond_dim != 1024 + mcfg.d_model:
        print(f"  WARN: ckpt conditioning_dim {cond_dim} != expected {1024 + mcfg.d_model}")
    model = BL3bModel(mcfg).to(device)
    # Load AR decoder weights
    ar_state = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
                for k, v in ckpt["ar_state"].items()}
    model.ar.load_state_dict(ar_state, strict=True)
    # Load LoRA adapter weights into the LoRA-wrapped V-JEPA. Base is frozen.
    lora_state = ckpt["lora_state"]
    missing, unexpected = model.vjepa.load_state_dict(lora_state, strict=False)
    if unexpected:
        print(f"  unexpected lora keys (ignored): {len(unexpected)}")
    return model


def main():
    ap = argparse.ArgumentParser(description="BL-3b standalone eval")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--emb-dir", default=str(EMB_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--n-samples-start", type=int, default=0)
    ap.add_argument("--n-samples-end", type=int, default=None)
    ap.add_argument("--stratified-target", type=int, default=None,
                    help="Stratified subset of approx N samples: ALL state-0 + "
                    "ALL state-2 + (N − that) from state-1.")
    ap.add_argument("--batch-size", type=int, default=4)   # smaller — V-JEPA fwd is heavy
    ap.add_argument("--warmup-calls", type=int, default=2)
    ap.add_argument("--save-raw", default=None,
                    help="Dump raw generated/reference seqs to this path; skips metrics.")
    ap.add_argument("--result-dir", default=str(RESULT_DIR))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    set_deterministic(args.seed)
    device = torch.device(args.device)

    sessions = _resolve_sessions(args)
    print(f"Evaluating BL-3b on {args.split}: {sessions}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    ecfg = EvalConfig(batch_size=args.batch_size)

    model = _load_bl3b_from_ckpt(ckpt, mcfg, device)
    model.eval()
    print(f"  Objective:     {ckpt.get('objective')}")
    train_p, total_p = model.param_count()
    print(f"  Train params:  {train_p:,}; total {total_p:,}")
    print(f"  Seed (ckpt):   {ckpt.get('seed')}; Torch (ckpt): {ckpt.get('torch_version')}")

    ds = _load_split_dataset(args, mcfg, sessions)
    dl = DataLoader(ds, batch_size=ecfg.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    print("\nComputing teacher-forced loss / perplexity...")
    tf_stats = _compute_tf_stats(model, dl, device, mcfg.vocab_size)
    print(f"  loss:    {tf_stats['tf_loss']:.4f}")
    print(f"  ppl:     {tf_stats['perplexity']:.2f}")
    print(f"  tf_acc:  {tf_stats['tf_token_acc']:.4f}")

    log_gpu_state(tag="pre-gen")

    all_reference, all_ref_lens, all_sensor_embs, all_fsm_states = _collect_references(
        dl, mcfg.vocab_size,
    )
    n_samples = len(all_reference)
    print(f"\nCollected {n_samples} reference traces (mean_len={np.mean(all_ref_lens):.1f})")
    print(f"  FSM-state distribution: {dict(sorted(Counter(all_fsm_states).items()))}")

    out: Dict = {
        "objective": ckpt.get("objective"),
        "split":      args.split,
        "checkpoint": str(args.checkpoint),
        "step":       ckpt.get("step", -1),
        "seed":       ckpt.get("seed"),
        "torch_version": ckpt.get("torch_version"),
        "sessions":   sessions,
        "n_samples":  n_samples,
        "mean_trace_len": float(np.mean(all_ref_lens)) if all_ref_lens else 0.0,
        "tf_loss":    tf_stats["tf_loss"],
        "perplexity": tf_stats["perplexity"],
        "token_acc_teacher_forced": tf_stats["tf_token_acc"],
    }

    raw_modes = {}
    for mode in ("greedy", "sampled"):
        print(f"\n=== Generation ({mode}) ===")
        generated, ms_list = _time_generation(
            model, dl, device, mcfg.max_seq_len, mode=mode, warmup_calls=args.warmup_calls,
        )
        generated = [g[:L] for g, L in zip(generated, all_ref_lens)]
        print(f"  Generated {len(generated)} samples")

        if args.save_raw:
            raw_modes[mode] = {"generated": generated, "ms_list": ms_list}
            continue

        print("  Computing metrics...")
        m = compute_generation_metrics(
            generated, all_reference, vocab_size=mcfg.vocab_size,
            fsm_states=all_fsm_states, seed=args.seed,
        )
        ts = summarize_ms(ms_list)
        denom = max(ecfg.batch_size * mcfg.max_seq_len, 1)
        ts_per_tick = {f"{k}_per_tick": v / denom for k, v in ts.items()
                       if isinstance(v, (int, float)) and k.endswith("_ms")}
        for k, v in m.items():
            out[f"{mode}_{k}"] = v
        for k, v in ts.items():
            out[f"{mode}_batch_{k}"] = v
        for k, v in ts_per_tick.items():
            out[f"{mode}_{k}"] = v
        print(f"    ned_mean    = {m['ned_mean']:.4f}")
        print(f"    exact_match = {m['exact_match']:.4f}")

    log_gpu_state(tag="post-gen")

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.save_raw:
        out_path = Path(args.save_raw)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "objective":     ckpt.get("objective"),
            "name":          "bl3b",
            "split":         args.split,
            "n_samples_start": int(args.n_samples_start),
            "n_samples_end":   int(args.n_samples_end if args.n_samples_end else args.n_samples),
            "references":      np.array(all_reference,  dtype=object),
            "ref_lens":        np.array(all_ref_lens,   dtype=np.int32),
            "fsm_states":      np.array(all_fsm_states, dtype=np.int32),
            "tf_loss":         tf_stats["tf_loss"],
            "perplexity":      tf_stats["perplexity"],
            "tf_token_acc":    tf_stats["tf_token_acc"],
            "torch_version":   ckpt.get("torch_version", ""),
            "checkpoint":      str(args.checkpoint),
            "sessions":        sessions,
            "sampling_steps":  -1,
            "batch_size":      ecfg.batch_size,
            "max_seq_len":     mcfg.max_seq_len,
        }
        for mode, raw in raw_modes.items():
            save_dict[f"{mode}_generated"] = np.array(raw["generated"], dtype=object)
            save_dict[f"{mode}_ms_list"]   = np.array(raw["ms_list"],   dtype=np.float64)
        np.savez_compressed(out_path, **save_dict)
        print(f"\nSaved RAW chunk -> {out_path}")
        return

    out_path = result_dir / f"bl3b_eval_{args.split}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
