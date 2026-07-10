"""Unified generator-objective evaluator.

Reads `ckpt["objective"]` to dispatch to MDLM / AR / AR-modern. Supports
`--baseline {unigram,copy_modal,copy_previous}` for trivial-floor eval that
goes through the same metrics pipeline as model eval — so floor and model
numbers are directly comparable.

Output schema is shared across variants (variant-specific fields may be
zero/null where not applicable). `perplexity` is NOT cross-variant
comparable (MDLM reports exp(ELBO), a bound; AR reports true PPL) — see
plan. Kept in per-variant JSONs for completeness; dropped from cross-
variant tables.

Usage:
    # Model eval (infers objective from checkpoint)
    python3 evaluate.py --checkpoint models/ar_fold_0/ar_best.pt \
        --split test --sessions s1 s2 --n-samples 1000

    # Trivial-baseline eval (no checkpoint needed; derives token distribution
    # from the training-fold sessions via --baseline-train-sessions)
    python3 evaluate.py --baseline copy_modal \
        --split test --sessions s1 s2 \
        --baseline-train-sessions t1 t2 \
        --result-dir results/ar_fold_0
"""
from __future__ import annotations

import os
# Must be set BEFORE any HF transformers / tokenizers import. Without this,
# DataLoader workers forking after tokenizer init can deadlock during MAUVE
# (HuggingFace prints "process just got forked, after parallelism has already
# been used" + the eval hangs forever — caused 6h timeouts on every BL-2/3/4/6
# eval in v4).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    ModelConfig, EvalConfig,
    VAL_SESSIONS, TEST_SESSIONS, EMB_DIR, DATA_DIR, RESULT_DIR,
)
from data import TraceDataset, load_token_mapping
from determinism import DEFAULT_SEED, set_deterministic
from metrics import compute_generation_metrics
from timing import gc_paused, log_gpu_state, summarize_ms, time_cuda, warmup

# Fixed seed for the eval-set subsample. Independent of --seed (which controls
# model sampling). Every variant compared under this harness must see the same
# held-out subset of ticks, so the subset RNG must NOT depend on user-supplied
# arguments. Bumping this number invalidates all prior result JSONs.
EVAL_SUBSET_SEED = 1234567


# ── Objective dispatch ──────────────────────────────────────────────────

def _build_model_and_sampler(ckpt: Dict[str, Any], mcfg: ModelConfig, device):
    """Return (model, sample_fn, objective_str) given a loaded checkpoint.

    sample_fn takes (sensor_emb, seq_len, *, greedy) → (B, seq_len) tokens.
    For MDLM, greedy flag maps to temperature=1.0 always (no separate greedy);
    we report only a single "sampled" pass for MDLM to keep schemas aligned.
    """
    objective = ckpt.get("objective")
    if objective is None:
        # Legacy MDLM checkpoints (pre-objective field).
        objective = "mdlm"

    if objective == "mdlm":
        from mdlm_model import MDLM
        model = MDLM(mcfg).to(device)

        def sample_fn(sensor_emb: torch.Tensor, seq_len: int, *,
                      greedy: bool = False,
                      n_steps: int = 1000,
                      temperature: float = 1.0) -> torch.Tensor:
            # MDLM has no distinct greedy mode; map greedy=True to temp≈0 via
            # argmax-like sharp posterior by using a very small temperature
            # (sample() clamps to temperature/1.0 anyway).
            temp = 1e-4 if greedy else temperature
            return model.sample(sensor_emb, seq_len=seq_len,
                                n_steps=n_steps, temperature=temp)

        return model, sample_fn, objective

    if objective == "autoregressive":
        from ar_model import ARModel
        model = ARModel(mcfg).to(device)

        def sample_fn(sensor_emb, seq_len, *, greedy=False, temperature=1.0, **_):
            return model.sample(sensor_emb, seq_len=seq_len,
                                temperature=temperature, greedy=greedy)

        return model, sample_fn, objective

    if objective == "autoregressive_modern":
        # Dispatch by ckpt flags. Three variants:
        # - state-token (E5): ARModernStateModel with state-prefix + CFG
        # - cross-attn (Tier 1A.1): ARModernCrossModel with Flamingo cross-attn
        # - prefix (legacy + Tier 2): ARModernModel
        use_state = bool(ckpt.get("use_state_token", False))
        use_cross = bool(ckpt.get("use_cross_attn", False))
        use_factored = bool(ckpt.get("use_factored_head", False))
        if use_state:
            from ar_modern_state_model import ARModernStateModel
            model = ARModernStateModel(mcfg).to(device)
        elif use_cross:
            from ar_modern_cross_model import ARModernCrossModel
            model = ARModernCrossModel(mcfg).to(device)
        elif use_factored:
            from ar_modern_factored_model import ARModernFactoredModel
            model = ARModernFactoredModel(mcfg).to(device)
        else:
            from ar_modern_model import ARModernModel
            model = ARModernModel(mcfg).to(device)

        def sample_fn(sensor_emb, seq_len, *, greedy=False, temperature=1.0,
                      guidance_scale=1.0, state_id=None, **_):
            kwargs = dict(seq_len=seq_len, temperature=temperature, greedy=greedy)
            if use_cross:
                kwargs["guidance_scale"] = guidance_scale
                return model.sample(sensor_emb, **kwargs)
            if use_state:
                if state_id is None:
                    raise ValueError("ARModernStateModel.sample requires state_id; "
                                     "ensure dataset provides state_pred")
                kwargs["guidance_scale"] = guidance_scale
                return model.sample(sensor_emb, state_id, **kwargs)
            return model.sample(sensor_emb, **kwargs)

        return model, sample_fn, objective

    if isinstance(objective, str) and objective.startswith("ar_modern_video_"):
        from ar_video_model import VideoConditionedARModern
        cond_dim = ckpt["conditioning_dim"]
        model = VideoConditionedARModern(mcfg, conditioning_dim=cond_dim).to(device)

        def sample_fn(conditioning, seq_len, *, greedy=False, temperature=1.0, **_):
            return model.sample(conditioning, seq_len=seq_len,
                                temperature=temperature, greedy=greedy)

        return model, sample_fn, objective

    raise ValueError(f"Unknown objective in checkpoint: {objective!r}")


# ── Reference / dataset loading ─────────────────────────────────────────

def _load_split_dataset(args, mcfg: ModelConfig, sessions: List[str],
                        video_flags: Optional[dict] = None):
    """Load full-session TraceDataset; if --n-samples is set, return a
    deterministic random Subset of size n_samples.

    For video objectives, the resulting dataset (Subset or full) is wrapped
    with VideoConditionedTraceDataset so each item carries vjepa_emb and/or
    cotracker_emb.

    Uses EVAL_SUBSET_SEED (not --seed): every variant compared under this
    harness sees the same held-out tick subset regardless of model-sampling
    seed. Smaller --n-samples is a strict prefix of larger --n-samples
    (one fixed permutation, take first N), so a 10K eval is contained in
    a 30K eval and metrics move monotonically with sample count."""
    if getattr(args, "joint_encoder_ckpt", None):
        # No-Perceiver / joint eval: raw sensor windows, encoded on the fly.
        from data import JointTraceWindowDataset
        ds = JointTraceWindowDataset(
            sessions, mcfg.max_seq_len, mcfg.vocab_size,
            window_size=args.bl7_window_size, data_dir=Path(args.data_dir),
            limit=None,
        )
    else:
        ds = TraceDataset(
            sessions, mcfg.max_seq_len, mcfg.vocab_size,
            emb_dir=Path(args.emb_dir), data_dir=Path(args.data_dir),
            limit=None,
        )
    # Subset selection: either deterministic random (default --n-samples N,
    # head of EVAL_SUBSET_SEED permutation) OR stratified-by-FSM-state.
    #
    # Stratified mode (--stratified-target N): take ALL state-0 + ALL state-2
    # + (N − that) from state-1. State-1 is sub-sampled via EVAL_SUBSET_SEED
    # permutation. Total ≈ N. Massively boosts minority-state sample counts
    # for paper-quality per-state confidence intervals; downstream collectors
    # should re-weight the "all" column by the natural FSM distribution
    # rather than the stratified mix.
    if args.stratified_target is not None and args.stratified_target > 0:
        fsm_arr = np.array(ds.fsm_states, dtype=np.int64)
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
        ds = torch.utils.data.Subset(ds, idx.tolist())
    elif args.n_samples is not None and args.n_samples > 0:
        rng = np.random.default_rng(EVAL_SUBSET_SEED)
        perm = rng.permutation(len(ds))
        if args.n_samples_end is not None and args.n_samples_end > 0:
            end = min(args.n_samples_end, len(perm), args.n_samples)
        else:
            end = min(args.n_samples, len(perm))
        start = max(0, args.n_samples_start)
        if start >= end:
            raise ValueError(f"empty chunk: start={start} end={end}")
        idx = np.sort(perm[start:end])           # parquet-row order for cache friendliness
        ds = torch.utils.data.Subset(ds, idx.tolist())
    needs_wrap = (video_flags is not None
                  and (video_flags.get("use_vjepa") or video_flags.get("use_cotracker")))
    if needs_wrap:
        from video_data import VideoConditionedTraceDataset
        ds = VideoConditionedTraceDataset(ds, **video_flags)
    return ds


def _collect_references(dl: DataLoader, pad_id: int
                        ) -> Tuple[List[List[int]], List[int], np.ndarray, List[int]]:
    """Walk the loader once, returning (sequences, trace_lens, sensor_embs, fsm_states)."""
    refs, lens, embs, fsm = [], [], [], []
    # Joint eval has no precomputed sensor_emb (raw windows are encoded on the
    # fly during generation); the collected embs are unused downstream, so we
    # skip them when only sensor_window is present.
    for batch in dl:
        x0 = batch["trace"]
        trace_len = batch["trace_len"]
        emb = batch.get("sensor_emb")
        fs = batch["fsm_state"]
        for i in range(x0.shape[0]):
            tl = int(trace_len[i])
            refs.append(x0[i, :tl].cpu().tolist())
            lens.append(tl)
            if emb is not None:
                embs.append(emb[i].cpu().numpy())
            fsm.append(int(fs[i]))
    return refs, lens, (np.stack(embs) if embs else np.zeros((0, 0))), fsm


# ── Trivial baselines ───────────────────────────────────────────────────

def _load_train_token_histogram(args, mcfg: ModelConfig,
                                train_sessions: List[str]) -> Counter:
    """Build a global token-frequency histogram from training sessions."""
    ds = TraceDataset(
        train_sessions, mcfg.max_seq_len, mcfg.vocab_size,
        emb_dir=Path(args.emb_dir), data_dir=Path(args.data_dir),
        limit=None,
    )
    hist: Counter = Counter()
    for i in range(len(ds)):
        item = ds[i]
        x0 = item["trace"]
        tl = int(item["trace_len"])
        for t in x0[:tl].tolist():
            if 0 <= t < mcfg.vocab_size:
                hist[t] += 1
    return hist


def _load_train_modal_traces(args, mcfg: ModelConfig,
                             train_sessions: List[str]) -> dict:
    """Most frequent full trace per FSM state in the training sessions.

    The per-token baselines are near-vacuous on this corpus (a trace is a
    sequence of ~91 distinct edges, so repeating one token yields NED≈0.97).
    The honest trivial floor is emitting the most common complete trace for
    the tick's FSM state, which exploits the corpus self-similarity the way
    a lookup table would.
    """
    ds = TraceDataset(
        train_sessions, mcfg.max_seq_len, mcfg.vocab_size,
        emb_dir=Path(args.emb_dir), data_dir=Path(args.data_dir),
        limit=None,
    )
    per_state: dict = {}
    for i in range(len(ds)):
        item = ds[i]
        tl = int(item["trace_len"])
        seq = tuple(item["trace"][:tl].tolist())
        s = int(item["fsm_state"])
        per_state.setdefault(s, Counter())[seq] += 1
    return {s: list(c.most_common(1)[0][0]) for s, c in per_state.items()}


def _baseline_generate(
    baseline: str,
    references: List[List[int]],
    trace_lens: List[int],
    token_hist: Counter,
    vocab_size: int,
    seed: int,
    fsm_states: List[int] = None,
    modal_traces: dict = None,
) -> List[List[int]]:
    """Produce a generated sequence per reference under the chosen floor."""
    rng = np.random.default_rng(seed)

    if baseline == "state_modal_trace":
        # Emit the training-set modal full trace for the tick's FSM state.
        assert fsm_states is not None and modal_traces is not None
        fallback = next(iter(modal_traces.values()))
        return [list(modal_traces.get(int(s), fallback)) for s in fsm_states]

    if baseline == "copy_modal":
        modal = token_hist.most_common(1)[0][0] if token_hist else 0
        return [[modal] * L for L in trace_lens]

    if baseline == "unigram":
        ids = np.fromiter(token_hist.keys(), dtype=np.int64)
        counts = np.fromiter(token_hist.values(), dtype=np.float64)
        probs = counts / counts.sum() if counts.sum() > 0 else None
        out = []
        for L in trace_lens:
            if probs is None:
                out.append(rng.integers(0, vocab_size, size=L).tolist())
            else:
                out.append(rng.choice(ids, size=L, replace=True, p=probs).tolist())
        return out

    if baseline == "copy_previous":
        # Seed with the modal token (the analog of BOS for this baseline);
        # then x[t] = x[t-1]. Equivalent to "always emit modal" for our
        # corpus, but we still emit it as a separate baseline for clarity.
        modal = token_hist.most_common(1)[0][0] if token_hist else 0
        out = []
        for ref in references:
            if not ref:
                out.append([])
                continue
            gen = [modal]
            for _ in range(1, len(ref)):
                gen.append(gen[-1])
            out.append(gen)
        return out

    raise ValueError(f"Unknown baseline: {baseline!r}")


# ── Inference timing ────────────────────────────────────────────────────

def _batch_conditioning(batch, device, video_flags=None):
    """Returns the conditioning vector to feed into sample_fn / compute_loss.

    For non-video objectives this is just sensor_emb (B, 512). For video
    variants, concat sensor_emb || vjepa_emb || cotracker_emb according to
    `video_flags = {"use_vjepa": ..., "use_cotracker": ...}`.
    """
    # Joint (no-Perceiver) eval feeds raw sensor windows; the model encodes
    # on the fly. JointTraceWindowDataset yields "sensor_window", not
    # "sensor_emb".
    if "sensor_window" in batch and "sensor_emb" not in batch:
        return batch["sensor_window"].to(device, non_blocking=True)
    if video_flags is None:
        return batch["sensor_emb"].to(device, non_blocking=True)
    from video_data import assemble_conditioning
    return assemble_conditioning(batch, **video_flags).to(device, non_blocking=True)


def _time_generation(
    sample_fn: Callable,
    dl: DataLoader,
    device,
    seq_len: int,
    mode: str,
    warmup_calls: int = 3,
    video_flags: Optional[dict] = None,
) -> Tuple[List[List[int]], List[float]]:
    """Generate samples with CUDA-event timing.

    Returns (generated, per_batch_elapsed_ms). `warmup_calls` warmup runs
    happen outside the timed loop to prime the cuDNN cache; every batch
    inside the loop is timed.
    """
    greedy = (mode == "greedy")
    generated: List[List[int]] = []
    ms_list: List[float] = []

    first_batch = None
    for batch in dl:
        first_batch = batch
        break
    if first_batch is not None:
        cond = _batch_conditioning(first_batch, device, video_flags)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            warmup(lambda: sample_fn(cond, seq_len, greedy=greedy), n=warmup_calls)

    with gc_paused():
        for batch_idx, batch in enumerate(dl):
            x0 = batch["trace"]
            cond = _batch_conditioning(batch, device, video_flags)
            trace_len = batch["trace_len"]

            def _run():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    return sample_fn(cond, seq_len, greedy=greedy)

            out, elapsed_ms = time_cuda(_run)
            ms_list.append(elapsed_ms)

            for i in range(x0.shape[0]):
                tl = int(trace_len[i])
                generated.append(out[i, :tl].cpu().tolist())

    return generated, ms_list


# ── Teacher-forced NLL (MDLM ELBO / AR true-PPL) ────────────────────────

def _compute_teacher_forced_stats(model, dl: DataLoader, device, objective: str,
                                   video_flags: Optional[dict] = None
                                  ) -> Dict[str, float]:
    """Walks the loader, computes mean loss + mean token accuracy under TF.

    For MDLM this is exp(ELBO); for AR it's true per-token PPL. Kept in the
    per-variant JSON but excluded from cross-variant comparison tables.
    """
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    mask_id = getattr(model, "mask_id", None)
    mask_or_pad = mask_id if mask_id is not None else getattr(model, "pad_id", 0)
    with torch.no_grad():
        for batch in dl:
            x0 = batch["trace"].to(device)
            cond = _batch_conditioning(batch, device, video_flags)
            pad_mask = (x0 != mask_or_pad).float()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, m = model.compute_loss(x0, cond, pad_mask)
            loss_sum += m["loss"]
            # AR exposes acc_teacher_forced; MDLM exposes acc_masked.
            acc_sum += m.get("acc_teacher_forced", m.get("acc_masked", 0.0))
            n += 1
    mean_loss = loss_sum / max(n, 1)
    mean_acc = acc_sum / max(n, 1)
    return {
        "tf_loss": mean_loss,
        "tf_token_acc": mean_acc,
        "perplexity": math.exp(min(mean_loss, 20)),
    }


# ── Main ────────────────────────────────────────────────────────────────

def _resolve_sessions(args) -> List[str]:
    if args.sessions:
        return args.sessions
    if args.split == "val":
        return VAL_SESSIONS
    if args.split == "test":
        return TEST_SESSIONS
    raise ValueError(f"Unknown split: {args.split}")


def _save_results(result_dir: Path, name: str, split: str,
                  results: Dict[str, Any]) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / f"{name}_eval_{split}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")
    return out_path


def run_model_eval(args) -> None:
    device = torch.device(args.device)
    sessions = _resolve_sessions(args)
    print(f"Evaluating on {args.split} split: {sessions}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mcfg_dict = ckpt["model_config"]
    mcfg = ModelConfig(**mcfg_dict)
    ecfg = EvalConfig(sampling_steps=args.sampling_steps, batch_size=args.batch_size)

    if getattr(args, "joint_encoder_ckpt", None):
        # No-Perceiver / joint eval: reconstruct JointEncoderARModel (frozen
        # 4000-tick encoder + trained cross-attn decoder) and encode raw
        # windows on the fly during sampling. No precomputed embeddings.
        from bl7_ksweep import build_ksweep_encoder, JointEncoderARModel
        from ar_modern_cross_model import ARModernCrossModel
        decoder = ARModernCrossModel(mcfg).to(device)
        encoder = build_ksweep_encoder(args.joint_encoder_ckpt, K=None, device=device)
        model = JointEncoderARModel(encoder, decoder, axial_mode=args.bl7_axial)
        objective = "autoregressive_modern"
        video_flags = None
        sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
              for k, v in ckpt["model"].items()}
        # Joint ckpt holds encoder.* (= frozen pretrained, already in `encoder`)
        # + ar.* (trained decoder). Loading the full dict restores both.
        model.load_state_dict(sd)
        print(f"  Joint no-Perceiver eval: encoder from {args.joint_encoder_ckpt}, "
              f"axial_mode={args.bl7_axial}, window={args.bl7_window_size}")

        def sample_fn(window, seq_len, *, greedy=False, temperature=1.0, **_):
            return model.sample(window, seq_len, temperature=temperature, greedy=greedy)

        model.eval()
    else:
        model, sample_fn, objective = _build_model_and_sampler(ckpt, mcfg, device)
        is_video_obj = isinstance(objective, str) and objective.startswith("ar_modern_video_")
        video_flags = ckpt.get("video_flags") if is_video_obj else None
        # If the checkpoint was trained on a different sensor encoder (e.g. BL-6
        # uses MOMENT embeddings cached at sensor_embeddings_moment/), override
        # the CLI default. The training script bakes the dir into the ckpt.
        if is_video_obj and "sensor_emb_dir" in ckpt and ckpt["sensor_emb_dir"]:
            args.emb_dir = ckpt["sensor_emb_dir"]
            print(f"  Using sensor emb dir from ckpt: {args.emb_dir}")

        state_dict = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
                      for k, v in ckpt["model"].items()}
        # E2 joint-trained checkpoints (JointEncoderARModel) store both encoder.*
        # and ar.* weights. The encoder weights are already baked into the
        # precomputed embeddings (--emb-dir), so for eval we load only the ar.*
        # subset into the standalone AR-modern decoder.
        is_joint_ckpt = any(k.startswith("ar.") for k in state_dict)
        if is_joint_ckpt:
            n_total = len(state_dict)
            state_dict = {k[len("ar."):]: v for k, v in state_dict.items()
                          if k.startswith("ar.")}
            print(f"  Joint checkpoint detected: loading {len(state_dict)}/{n_total} "
                  f"ar.* keys (encoder.* weights are baked into --emb-dir)")
        model.load_state_dict(state_dict)

        # Apply EMA weights if available. Skip for joint checkpoints (EMA tracks
        # the full encoder+ar params and doesn't align with the standalone AR).
        if "ema" in ckpt and ckpt["ema"] is not None and not is_joint_ckpt:
            from mdlm_train import EMA
            ema = EMA(model.parameters(), decay=ckpt["ema"]["decay"])
            ema.load_state_dict(ckpt["ema"])
            ema.to(device)
            ema.copy_to(model.parameters())
            print("  Applied EMA weights")
        elif is_joint_ckpt:
            print("  Joint checkpoint: skipping EMA apply (using raw ckpt['model'] "
                  "ar.* weights — consistent with the extracted encoder embeddings)")

    model.eval()
    print(f"  Objective:     {objective}")
    print(f"  Model params:  {model.param_count():,}")
    print(f"  Seed (ckpt):   {ckpt.get('seed')}")
    print(f"  Torch (ckpt):  {ckpt.get('torch_version')}")

    ds = _load_split_dataset(args, mcfg, sessions, video_flags=video_flags)
    dl = DataLoader(ds, batch_size=ecfg.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # Teacher-forced likelihood / token accuracy
    print("\nComputing teacher-forced loss / perplexity...")
    tf_stats = _compute_teacher_forced_stats(model, dl, device, objective,
                                              video_flags=video_flags)
    print(f"  loss:      {tf_stats['tf_loss']:.4f}")
    print(f"  ppl:       {tf_stats['perplexity']:.2f}")
    print(f"  tf_acc:    {tf_stats['tf_token_acc']:.4f}")

    log_gpu_state(tag="pre-gen")

    # AR has a separate greedy/sampled distinction; MDLM only sampled.
    is_ar = (objective in {"autoregressive", "autoregressive_modern"}
             or (isinstance(objective, str) and objective.startswith("ar_modern_video_")))
    if getattr(args, "greedy_only", False) and is_ar:
        modes = ["greedy"]            # skip the sampled pass (halves gen time)
    else:
        modes = (["greedy", "sampled"] if is_ar else ["sampled"])

    all_reference, all_ref_lens, all_sensor_embs, all_fsm_states = \
        _collect_references(dl, mcfg.vocab_size)
    n_samples = len(all_reference)
    print(f"\nCollected {n_samples} reference traces (mean_len={np.mean(all_ref_lens):.1f})")
    fsm_counts = Counter(all_fsm_states)
    print(f"  FSM-state distribution: {dict(sorted(fsm_counts.items()))}")

    out: Dict[str, Any] = {
        "objective": objective,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "step": ckpt.get("step", -1),
        "seed": ckpt.get("seed"),
        "torch_version": ckpt.get("torch_version"),
        "sessions": sessions,
        "n_samples": n_samples,
        "sampling_steps": ecfg.sampling_steps if objective == "mdlm" else None,
        "mean_trace_len": float(np.mean(all_ref_lens)) if all_ref_lens else 0.0,
        "tf_loss": tf_stats["tf_loss"],
        "perplexity": tf_stats["perplexity"],
        "token_acc_teacher_forced": tf_stats["tf_token_acc"],
    }

    raw_modes: Dict[str, Any] = {}
    for mode in modes:
        print(f"\n=== Generation ({mode}) ===")
        generated, ms_list = _time_generation(
            sample_fn, dl, device, mcfg.max_seq_len, mode=mode,
            warmup_calls=args.warmup_calls,
            video_flags=video_flags,
        )
        # Crop generated to reference length for fair metric comparison.
        generated = [g[:L] for g, L in zip(generated, all_ref_lens)]
        print(f"  Generated {len(generated)} samples")

        if args.save_raw:
            raw_modes[mode] = {"generated": generated, "ms_list": ms_list}
            continue        # skip metric computation; merge step does it on full set

        print(f"  Computing metrics...")
        m = compute_generation_metrics(
            generated, all_reference, vocab_size=mcfg.vocab_size,
            fsm_states=all_fsm_states,
            seed=args.seed,
        )
        # Per-batch ms summary
        ts = summarize_ms(ms_list)
        # Approx ms/tick: median batch_ms / (batch_size * seq_len).
        denom = max(ecfg.batch_size * mcfg.max_seq_len, 1)
        ts_per_tick = {f"{k}_per_tick": v / denom for k, v in ts.items()
                       if isinstance(v, (int, float)) and k.endswith("_ms")}

        prefix = mode  # e.g. "greedy" or "sampled"
        for k, v in m.items():
            out[f"{prefix}_{k}"] = v
        for k, v in ts.items():
            out[f"{prefix}_batch_{k}"] = v
        for k, v in ts_per_tick.items():
            out[f"{prefix}_{k}"] = v

        # Log the first few numbers for sanity
        print(f"    ned_mean            = {m['ned_mean']:.4f}")
        print(f"    exact_match         = {m['exact_match']:.4f}")
        print(f"    vendi_ratio         = {m.get('vendi_ratio_gen_over_ref', 0):.4f}")
        per_state = m.get("per_state", {})
        for s in sorted(per_state, key=int):
            d = per_state[s]
            print(f"    state {s}             n={d['n']} ned={d['ned_mean']:.4f} "
                  f"exact={d['exact_match']:.4f} vendi_ratio={d['vendi_ratio_gen_over_ref']:.4f}")
        print(f"    crystal_bleu        = {m['crystal_bleu']:.4f}")
        print(f"    median_ms/tick      = {ts_per_tick.get('median_ms_per_tick', 0):.4f}")

    log_gpu_state(tag="post-gen")

    # Filename: objective-prefixed for unique JSON per model variant.
    name_map = {
        "mdlm": "mdlm",
        "autoregressive": "ar",
        "autoregressive_modern": "ar_modern",
    }
    if objective.startswith("ar_modern_video_"):
        name = objective.replace("ar_modern_video_", "")     # bl3 / bl4 / bl5 / bl6 / bl6b
    else:
        name = name_map[objective]

    result_dir = Path(args.result_dir) if args.result_dir else RESULT_DIR

    if args.save_raw:
        # Dump raw sequences + timing for every mode; merge_chunks.py recomputes
        # metrics on the concatenated full set.
        out_path = Path(args.save_raw)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "objective": objective,
            "name":      name,
            "split":     args.split,
            "n_samples_start": int(args.n_samples_start),
            "n_samples_end":   int(args.n_samples_end if args.n_samples_end else args.n_samples),
            "references":      np.array(all_reference, dtype=object),
            "ref_lens":        np.array(all_ref_lens,  dtype=np.int32),
            "fsm_states":      np.array(all_fsm_states, dtype=np.int32),
            "tf_loss":         tf_stats["tf_loss"],
            "perplexity":      tf_stats["perplexity"],
            "tf_token_acc":    tf_stats["tf_token_acc"],
            "torch_version":   ckpt.get("torch_version", ""),
            "checkpoint":      str(args.checkpoint),
            "sessions":        sessions,
            "sampling_steps":  ecfg.sampling_steps if objective == "mdlm" else -1,
            "batch_size":      ecfg.batch_size,
            "max_seq_len":     mcfg.max_seq_len,
        }
        for mode, raw in raw_modes.items():
            save_dict[f"{mode}_generated"] = np.array(raw["generated"], dtype=object)
            save_dict[f"{mode}_ms_list"]   = np.array(raw["ms_list"],   dtype=np.float64)
        np.savez_compressed(out_path, **save_dict)
        print(f"\nSaved RAW chunk -> {out_path} (no metrics computed; merge_chunks.py finishes)")
        return

    _save_results(result_dir, name=name, split=args.split, results=out)


def run_baseline_eval(args) -> None:
    """Trivial-floor eval through the same metrics pipeline as model eval."""
    sessions = _resolve_sessions(args)
    print(f"Baseline '{args.baseline}' on {args.split} split: {sessions}")

    # Minimal ModelConfig — we only need max_seq_len + vocab_size for the
    # dataset. Use the project defaults (config module).
    mcfg = ModelConfig()
    vocab_size, _, _ = load_token_mapping()
    # Trust the token mapping over the dataclass default.
    mcfg = ModelConfig(vocab_size=vocab_size, mask_token_id=vocab_size)

    ds = _load_split_dataset(args, mcfg, sessions)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    references, trace_lens, sensor_embs, fsm_states = \
        _collect_references(dl, mcfg.vocab_size)
    n_samples = len(references)
    print(f"  Collected {n_samples} reference traces")
    fsm_counts = Counter(fsm_states)
    print(f"  FSM-state distribution: {dict(sorted(fsm_counts.items()))}")

    train_sessions = args.baseline_train_sessions or sessions
    print(f"  Building token histogram from: {train_sessions}")
    hist = _load_train_token_histogram(args, mcfg, train_sessions)
    modal = hist.most_common(1)[0] if hist else (None, 0)
    print(f"  Histogram: {len(hist)} unique tokens; modal={modal}")

    modal_traces = None
    if args.baseline == "state_modal_trace":
        print("  Building per-state modal traces from training sessions...")
        modal_traces = _load_train_modal_traces(args, mcfg, train_sessions)
        for s, seq in sorted(modal_traces.items()):
            print(f"    state {s}: modal trace len={len(seq)}")

    generated = _baseline_generate(args.baseline, references, trace_lens,
                                   hist, mcfg.vocab_size, seed=args.seed,
                                   fsm_states=fsm_states,
                                   modal_traces=modal_traces)

    print("  Computing metrics...")
    m = compute_generation_metrics(generated, references,
                                   vocab_size=mcfg.vocab_size,
                                   fsm_states=fsm_states,
                                   seed=args.seed)

    out = {
        "objective": f"baseline/{args.baseline}",
        "split": args.split,
        "sessions": sessions,
        "baseline_train_sessions": train_sessions,
        "n_samples": n_samples,
        "mean_trace_len": float(np.mean(trace_lens)) if trace_lens else 0.0,
    }
    for k, v in m.items():
        out[f"baseline_{k}"] = v

    print(f"    ned_mean            = {m['ned_mean']:.4f}")
    print(f"    exact_match         = {m['exact_match']:.4f}")
    print(f"    vendi_ratio         = {m.get('vendi_ratio_gen_over_ref', 0):.4f}")
    per_state = m.get("per_state", {})
    for s in sorted(per_state, key=int):
        d = per_state[s]
        print(f"    state {s}             n={d['n']} ned={d['ned_mean']:.4f} "
              f"exact={d['exact_match']:.4f} vendi_ratio={d['vendi_ratio_gen_over_ref']:.4f}")
    print(f"    crystal_bleu        = {m['crystal_bleu']:.4f}")

    result_dir = Path(args.result_dir) if args.result_dir else RESULT_DIR
    _save_results(result_dir, name=args.baseline, split=args.split, results=out)


def main():
    ap = argparse.ArgumentParser(description="Unified generator-objective evaluator")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to model checkpoint. Omit when --baseline is set.")
    ap.add_argument("--baseline", default=None,
                    choices=["unigram", "copy_modal", "copy_previous",
                             "state_modal_trace"],
                    help="If set, run trivial-baseline eval instead of model eval.")
    ap.add_argument("--baseline-train-sessions", nargs="+", default=None,
                    help="Sessions used to build the token histogram for unigram / "
                         "copy_modal / copy_previous. Defaults to eval sessions (fine "
                         "for smoke tests; use the fold's train sessions for real runs).")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--emb-dir", default=str(EMB_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers. 0 (default) avoids fork-after-init "
                    "deadlocks during MAUVE/HF-tokenizer-derived metrics. With "
                    "MAUVE dropped, 0 also has no perf cost on our small-batch eval.")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Total subset size (deterministic via EVAL_SUBSET_SEED). "
                    "When chunking, also pass --n-samples-start/--n-samples-end.")
    ap.add_argument("--n-samples-start", type=int, default=0,
                    help="Inclusive start index of permutation slice [start:end). "
                    "Default 0 reproduces full-eval behavior.")
    ap.add_argument("--n-samples-end", type=int, default=None,
                    help="Exclusive end index. If unset, uses --n-samples.")
    ap.add_argument("--stratified-target", type=int, default=None,
                    help="If set, build a stratified subset of approx N samples: "
                    "ALL of state-0 + ALL of state-2 + (N − that) from state-1. "
                    "Massively boosts minority-state sample counts for "
                    "paper-quality per-state CIs. Combine with --n-samples-start/end "
                    "for chunked eval. Mutually exclusive with --n-samples (random).")
    ap.add_argument("--save-raw", default=None,
                    help="When set, dump raw generated/reference sequences + "
                    "fsm_states + per-batch timing to this .npz path INSTEAD of "
                    "computing metrics. Used for chunked evals (merge_chunks.py "
                    "concatenates the chunks then computes metrics on the full set).")
    ap.add_argument("--sampling-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--warmup-calls", type=int, default=3)
    ap.add_argument("--greedy-only", action="store_true",
                    help="Skip the sampled generation pass; only greedy "
                         "(halves gen time; greedy NED is the comparison metric).")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="Override sessions to evaluate on")
    ap.add_argument("--result-dir", default=None,
                    help="Override result output directory")
    # No-Perceiver / joint eval: encode raw sensor windows on the fly with a
    # frozen encoder + trained cross-attn decoder (no precomputed embeddings).
    ap.add_argument("--joint-encoder-ckpt", default=None,
                    help="Path to the frozen encoder pretrain checkpoint (e.g. "
                         "models/bl7_w4000/bl7_best.pt). When set, evaluate.py "
                         "builds a JointEncoderARModel and encodes raw windows "
                         "on the fly instead of loading --emb-dir.")
    ap.add_argument("--bl7-axial", action="store_true",
                    help="Joint eval: use the no-Perceiver axial token path "
                         "(decoder cross-attends to the full pre-pool tokens).")
    ap.add_argument("--bl7-window-size", type=int, default=500,
                    help="Joint eval: raw sensor-window length for the dataset "
                         "(must match the encoder checkpoint's window_size).")
    args = ap.parse_args()

    set_deterministic(args.seed)

    if args.baseline is not None:
        run_baseline_eval(args)
    else:
        if args.checkpoint is None:
            ap.error("Provide either --checkpoint or --baseline.")
        run_model_eval(args)


if __name__ == "__main__":
    main()
