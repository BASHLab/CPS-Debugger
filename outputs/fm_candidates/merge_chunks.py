"""Merge eval-chunk .npz files into a single per-(variant, fold, split) JSON.

Each chunk is produced by `evaluate.py --save-raw <path>` running on a
deterministic permutation slice [start:end). The chunks are non-overlapping
and tile the full --n-samples count exactly. We concatenate the raw
generated/reference sequences across chunks and call
`compute_generation_metrics` on the union — this avoids per-metric merge
arithmetic (Vendi, Self-BLEU, distinct-n etc. all want the full set).

Per-batch ms timings from each chunk are pooled into one list before the
median/p90/p99 summary so cross-chunk timing is comparable.

Usage:
    python3 merge_chunks.py \\
        --chunks results/bl3_fold_0/bl3_eval_test_chunk_*.npz \\
        --result-dir results/bl3_fold_0 \\
        --name bl3 --split test
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

from config import EvalConfig
from metrics import compute_generation_metrics
from timing import summarize_ms


def _seqs_from_obj(arr) -> list:
    """object-dtype npz array of variable-length lists -> list of lists."""
    return [list(seq) for seq in arr.tolist()]


def merge_chunks(chunk_paths, result_dir: Path, name: str, split: str,
                 vocab_size: int, max_seq_len: int):
    chunks = []
    for p in sorted(chunk_paths):
        with np.load(p, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        chunks.append((p, data))
    if not chunks:
        raise SystemExit("no chunks supplied")

    # Sanity: every chunk should agree on objective, name, split, sampling cfg.
    head = chunks[0][1]
    for p, d in chunks[1:]:
        for key in ("objective", "name", "split", "sampling_steps", "batch_size", "max_seq_len"):
            if str(d[key]) != str(head[key]):
                raise SystemExit(f"chunk mismatch at {p} on '{key}': "
                                 f"{d[key]!r} vs head {head[key]!r}")

    # Concatenate raw sequences across chunks (in chunk-start order).
    all_refs = []
    all_ref_lens = []
    all_fsm = []
    for _, d in chunks:
        all_refs.extend(_seqs_from_obj(d["references"]))
        all_ref_lens.extend(d["ref_lens"].tolist())
        all_fsm.extend(d["fsm_states"].tolist())

    # Detect generation modes by suffix in keys
    modes = sorted({k[:-len("_generated")] for k in head.keys()
                    if k.endswith("_generated")})
    print(f"Modes: {modes}")
    print(f"Total samples after merge: {len(all_refs)}")

    out = {
        "objective": str(head["objective"]),
        "split": split,
        "checkpoint": str(head["checkpoint"]),
        "sessions": [str(s) for s in head["sessions"].tolist()],
        "n_samples": len(all_refs),
        "sampling_steps": int(head["sampling_steps"]) if int(head["sampling_steps"]) > 0 else None,
        "mean_trace_len": float(np.mean(all_ref_lens)) if all_ref_lens else 0.0,
        "tf_loss": float(np.mean([float(d["tf_loss"]) for _, d in chunks])),
        "perplexity": float(np.mean([float(d["perplexity"]) for _, d in chunks])),
        "token_acc_teacher_forced": float(np.mean(
            [float(d["tf_token_acc"]) for _, d in chunks])),
        "torch_version": str(head["torch_version"]),
        "merged_from_chunks": [str(p) for p, _ in chunks],
    }

    ecfg = EvalConfig()
    ecfg.batch_size = int(head["batch_size"])

    for mode in modes:
        # Concat generated sequences across chunks
        all_gen = []
        all_ms  = []
        for _, d in chunks:
            all_gen.extend(_seqs_from_obj(d[f"{mode}_generated"]))
            all_ms.extend(d[f"{mode}_ms_list"].tolist())
        # Crop to ref length (idempotent, but match existing pipeline)
        all_gen = [g[:L] for g, L in zip(all_gen, all_ref_lens)]

        m = compute_generation_metrics(
            all_gen, all_refs, vocab_size=vocab_size, fsm_states=all_fsm, seed=0,
        )
        ts = summarize_ms(all_ms)
        denom = max(ecfg.batch_size * max_seq_len, 1)
        ts_per_tick = {f"{k}_per_tick": v / denom for k, v in ts.items()
                       if isinstance(v, (int, float)) and k.endswith("_ms")}
        for k, v in m.items():
            out[f"{mode}_{k}"] = v
        for k, v in ts.items():
            out[f"{mode}_batch_{k}"] = v
        for k, v in ts_per_tick.items():
            out[f"{mode}_{k}"] = v
        print(f"[{mode}] ned_mean={m['ned_mean']:.4f} exact={m['exact_match']:.4f} "
              f"vendi_ratio={m.get('vendi_ratio_gen_over_ref', 0):.4f} "
              f"crystal={m['crystal_bleu']:.4f}")

    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / f"{name}_eval_{split}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True,
                    help="Glob pattern(s) or explicit paths to chunk .npz files.")
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--name", required=True,
                    help="Variant short name used in JSON filename "
                    "(mdlm / ar_modern / bl3 / bl4 / bl6).")
    ap.add_argument("--split", required=True, choices=["test", "val"])
    ap.add_argument("--vocab-size", type=int, default=648)
    ap.add_argument("--max-seq-len", type=int, default=128)
    args = ap.parse_args()

    chunk_paths = []
    for pat in args.chunks:
        if "*" in pat or "?" in pat or "[" in pat:
            chunk_paths.extend(glob.glob(pat))
        else:
            chunk_paths.append(pat)
    chunk_paths = sorted(set(chunk_paths))

    merge_chunks(chunk_paths, Path(args.result_dir), args.name, args.split,
                 args.vocab_size, args.max_seq_len)


if __name__ == "__main__":
    main()
