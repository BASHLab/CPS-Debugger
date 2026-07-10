#!/usr/bin/env python3
"""ASPK Binary Trace Format EDA.

Decodes raw .aspk execution trace files and characterizes them for FM
sequence-generation architecture decisions.

Format (verified against processed_trace cf_table):
  8-byte header (write-call self-trace, ignored)
  N x 8-byte entries: (to_pc: u24 LE, from_pc: u24 LE, func_id: u16 LE)
"""
import csv, json, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────
SESSION = "2025-03-18_12-39-10"
EXTRACTED = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted")
SESS = EXTRACTED / SESSION
TRACE_DIR = SESS / "trace"
PT_PATH = SESS / "processed_trace" / "processed_trace.csv"
DL_PATH = SESS / "datalayer" / "datalayer.csv"

EDA_DIR = Path("/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_trace_eda")
OUT = EDA_DIR / "aspk_eda"
OUT.mkdir(exist_ok=True)
REPORT_PATH = OUT / "aspk_eda_report.md"

# ── constants ──────────────────────────────────────────────────────────
CONTROLLER_FUNCS = {20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46}
MATH_FUNCS = {63,64,70,71,72,73,74,77,78,99}
CTRL_MATH = CONTROLLER_FUNCS | MATH_FUNCS

# ── report accumulator ────────────────────────────────────────────────
_R = []
def R(s=""):
    _R.append(s)
    print(s)

def save_report():
    REPORT_PATH.write_text("\n".join(_R) + "\n")

# ── helpers ────────────────────────────────────────────────────────────
def load_func_names():
    with open(EDA_DIR / "func_id_to_name.json") as f:
        return {int(k): v for k, v in json.load(f).items()}

def load_inventory():
    return pd.read_csv(EDA_DIR / "all_edges_inventory.csv")

def parse_aspk_filename(name):
    # cc:<cc>_ts:<ts>.aspk
    base = name[:-len(".aspk")] if name.endswith(".aspk") else name
    parts = base.split("_ts:")
    cc = int(parts[0].split(":",1)[1])
    ts = int(parts[1])
    return cc, ts

def decode_aspk(data):
    """Returns list of (func_id, from_pc, to_pc) ordered. Skips 8-byte header."""
    entries = []
    n = len(data)
    for i in range(8, n - 7, 8):
        chunk = data[i:i+8]
        to_pc   = chunk[0] | (chunk[1] << 8) | (chunk[2] << 16)
        from_pc = chunk[3] | (chunk[4] << 8) | (chunk[5] << 16)
        func    = chunk[6] | (chunk[7] << 8)
        entries.append((func, from_pc, to_pc))
    return entries

def list_aspk_sorted():
    """Return list of (cc, ts, path) sorted by cc."""
    items = []
    for f in TRACE_DIR.iterdir():
        if not f.name.endswith(".aspk"): continue
        try:
            cc, ts = parse_aspk_filename(f.name)
        except Exception:
            continue
        items.append((cc, ts, f))
    items.sort(key=lambda x: x[0])
    return items

def parse_cf_table(s):
    cf = json.loads(s)
    out = Counter()
    for fid_s, fd in cf.items():
        fid = int(fid_s)
        for fp_s, td in fd.items():
            fp = int(fp_s)
            for tp_s, c in td.items():
                out[(fid, fp, int(tp_s))] = c
    return out

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — Binary format verification
# ══════════════════════════════════════════════════════════════════════
def section_1(all_files, func_names):
    R("\n" + "="*80)
    R("# Section 1: Binary Format Verification")
    R("="*80)

    R(f"\nFormat hypothesis: 8-byte header + N x 8-byte entries")
    R(f"  Each entry: bytes[0:3]=to_pc (u24 LE), bytes[3:6]=from_pc (u24 LE), bytes[6:8]=func_id (u16 LE)")

    # Pick 10 files of varying sizes
    sizes = [(p.stat().st_size, cc, ts, p) for cc, ts, p in all_files[::len(all_files)//200]]
    sizes.sort()
    picks = [sizes[0], sizes[len(sizes)//4], sizes[len(sizes)//2],
             sizes[3*len(sizes)//4], sizes[-1]]
    # Add 5 more from the middle
    picks += [sizes[len(sizes)//8], sizes[3*len(sizes)//8],
              sizes[5*len(sizes)//8], sizes[7*len(sizes)//8],
              sizes[len(sizes)-2]]
    picks = picks[:10]

    R("\n10 sample files (size, decoded entries):")
    R(f"  {'size_B':>8} {'cc':>10} {'ts':>20}  n_entries  divisible_by_8")
    R("  " + "-"*70)
    for sz, cc, ts, p in picks:
        data = p.read_bytes()
        ents = decode_aspk(data)
        ok = (sz - 8) % 8 == 0
        R(f"  {sz:>8} {cc:>10} {ts:>20}  {len(ents):>9}  {ok}")

    # Cross-reference verification: sample one PT row, find matching aspk, compare counts
    R("\n--- Cross-reference verification against processed_trace ---")
    n_verified = 0
    n_perfect = 0
    with open(PT_PATH) as f:
        reader = csv.DictReader(f)
        ts_to_path = {ts: p for cc, ts, p in all_files}
        for i, row in enumerate(reader):
            ts_us = int(row["timestamp"])
            if ts_us not in ts_to_path:
                continue
            cf = parse_cf_table(row["cf_table"])
            data = ts_to_path[ts_us].read_bytes()
            ents = decode_aspk(data)
            cnt = Counter(ents)
            tot_match = sum(cnt[t] for t in cnt if cf.get(t,0) == cnt[t])
            tot_pt = sum(cf.values())
            tot_aspk = sum(cnt.values())
            unique_match = sum(1 for t in cnt if cf.get(t,0) == cnt[t])
            n_verified += 1
            if cnt == cf:
                n_perfect += 1
            if n_verified <= 5:
                R(f"  ts={ts_us}  PT_total={tot_pt}  aspk_total={tot_aspk}  "
                  f"unique={len(cnt)}  exact_match_unique={unique_match}/{len(cnt)}  "
                  f"perfect_dict_eq={cnt == cf}")
            if n_verified >= 20:
                break
    R(f"\n  Verified {n_verified} aspk files against processed_trace, {n_perfect} perfect matches.")

    # Print 3 fully decoded example ticks
    R("\n--- 3 fully decoded example ticks ---")
    samp_picks = [all_files[len(all_files)//10],
                  all_files[len(all_files)//2],
                  all_files[9*len(all_files)//10]]
    for cc, ts, p in samp_picks:
        data = p.read_bytes()
        ents = decode_aspk(data)
        R(f"\n  cc={cc} ts={ts} size={len(data)}B  n_entries={len(ents)}")
        R(f"  First 12 entries (func_id, func_name, from_pc -> to_pc):")
        for j, (f,fp,tp) in enumerate(ents[:12]):
            fname = func_names.get(f, "?")
            R(f"    [{j:3d}]  func={f:>3} {fname:25s}  {fp:>6} -> {tp:<6}")
        if len(ents) > 12:
            R(f"    ... ({len(ents)-12} more)")

# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — Sequence length distribution (10k uniform samples)
# ══════════════════════════════════════════════════════════════════════
def section_2(all_files, func_names):
    R("\n" + "="*80)
    R("# Section 2: Sequence Length Distribution (10,000 uniform samples)")
    R("="*80)

    n = len(all_files)
    n_sample = min(10000, n)
    # Uniform stride
    idxs = np.linspace(0, n-1, n_sample, dtype=int)
    samples = [all_files[i] for i in idxs]

    rows = []
    t0 = time.time()
    for j, (cc, ts, p) in enumerate(samples):
        data = p.read_bytes()
        ents = decode_aspk(data)
        ctrl_n = sum(1 for f,_,_ in ents if f in CTRL_MATH)
        lib_n = len(ents) - ctrl_n
        rows.append({"cc": cc, "ts_us": ts, "size_B": len(data),
                     "n_entries": len(ents), "n_ctrl_math": ctrl_n, "n_library": lib_n})
        if (j+1) % 2000 == 0:
            print(f"    sampled {j+1}/{n_sample}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "seq_length_dist.csv", index=False)
    R(f"\nSaved {len(df)} samples to seq_length_dist.csv")

    def stats(col, label):
        v = df[col].values
        R(f"\n  {label}:")
        R(f"    n={len(v)}  min={v.min()}  max={v.max()}  mean={v.mean():.1f}  median={int(np.median(v))}  std={v.std():.1f}")
        R(f"    p5={int(np.percentile(v,5))}  p25={int(np.percentile(v,25))}  "
          f"p75={int(np.percentile(v,75))}  p95={int(np.percentile(v,95))}  p99={int(np.percentile(v,99))}")

    stats("n_entries", "ALL entries per tick")
    stats("n_ctrl_math", "Controller+Math entries per tick")
    stats("n_library", "Library entries per tick")

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["n_entries"], bins=80, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("entries per tick")
    axes[0].set_ylabel("count (log)")
    axes[0].set_title("All entries / tick")
    axes[1].hist(df["n_ctrl_math"], bins=80, color="darkorange", edgecolor="black", alpha=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("ctrl+math entries per tick")
    axes[1].set_ylabel("count (log)")
    axes[1].set_title("Controller+Math entries / tick")
    plt.tight_layout()
    plt.savefig(OUT / "fig_seq_length_aspk.png", dpi=110)
    plt.close()
    R(f"\n  Saved histogram → fig_seq_length_aspk.png")

    return df, samples

# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — Vocabulary
# ══════════════════════════════════════════════════════════════════════
def section_3(samples, func_names):
    R("\n" + "="*80)
    R("# Section 3: Edge Triple Vocabulary")
    R("="*80)

    vocab = Counter()
    for cc, ts, p in samples:
        data = p.read_bytes()
        ents = decode_aspk(data)
        for e in ents:
            vocab[e] += 1

    n_total = len(vocab)
    n_ctrl = sum(1 for (f,_,_) in vocab if f in CTRL_MATH)
    n_lib = n_total - n_ctrl
    sum_ctrl = sum(c for (f,_,_),c in vocab.items() if f in CTRL_MATH)
    sum_lib = sum(c for (f,_,_),c in vocab.items() if f not in CTRL_MATH)

    R(f"\n  Unique edge triples: {n_total}")
    R(f"    Controller+Math: {n_ctrl}  (raw count: {sum_ctrl})")
    R(f"    Library:         {n_lib}  (raw count: {sum_lib})")
    R(f"  Token vocabulary size if generating triples directly: {n_total}")

    # Save vocabulary
    rows = []
    for (f,fp,tp), c in vocab.most_common():
        rows.append({"func_id": f, "func_name": func_names.get(f,"?"),
                     "from_pc": fp, "to_pc": tp, "count": c,
                     "category": "ctrl_math" if f in CTRL_MATH else "library"})
    pd.DataFrame(rows).to_csv(OUT / "edge_vocabulary.csv", index=False)
    R(f"\n  Saved vocabulary → edge_vocabulary.csv")

    R("\n  Top 20 most frequent edge triples:")
    R(f"    {'rank':>4} {'func':>4} {'func_name':25s} {'from':>6}->{'to':<6}  {'count':>10}  cat")
    R("    " + "-"*70)
    for i, ((f,fp,tp), c) in enumerate(vocab.most_common(20), 1):
        cat = "ctrl" if f in CTRL_MATH else "lib"
        R(f"    {i:>4} {f:>4} {func_names.get(f,'?'):25s} {fp:>6}->{tp:<6}  {c:>10}  {cat}")

    return vocab

# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — Timestamp alignment
# ══════════════════════════════════════════════════════════════════════
def section_4(all_files):
    R("\n" + "="*80)
    R("# Section 4: Timestamp Alignment to Datalayer")
    R("="*80)

    # Load datalayer
    dl = pd.read_csv(DL_PATH)
    dl["ts_us"] = (dl["timestamp"] // 1000).astype(np.int64)
    dl = dl.sort_values("ts_us").reset_index(drop=True)
    R(f"\n  Datalayer: {len(dl)} rows")
    R(f"    ts_us range: {dl['ts_us'].iloc[0]} … {dl['ts_us'].iloc[-1]}")

    # Sample 100 aspk files uniformly
    n = len(all_files)
    idxs = np.linspace(0, n-1, 100, dtype=int)
    samples = [all_files[i] for i in idxs]

    dl_ts = dl["ts_us"].values
    deltas = []
    examples = []
    for cc, ts, p in samples:
        idx = np.searchsorted(dl_ts, ts)
        if idx >= len(dl_ts): idx = len(dl_ts) - 1
        if idx > 0 and abs(dl_ts[idx-1] - ts) < abs(dl_ts[idx] - ts):
            idx -= 1
        d = abs(int(dl_ts[idx]) - ts)
        deltas.append(d)
        if len(examples) < 3:
            examples.append((ts, int(dl_ts[idx]), idx, cc))

    deltas = np.array(deltas)
    R(f"\n  Aligned 100 aspk files:")
    R(f"    mean alignment error = {deltas.mean():.1f} µs")
    R(f"    max  alignment error = {deltas.max()} µs")
    R(f"    std  alignment error = {deltas.std():.1f} µs")
    R(f"    p50/p95/p99 = {int(np.percentile(deltas,50))} / "
      f"{int(np.percentile(deltas,95))} / {int(np.percentile(deltas,99))} µs")

    R("\n  3 alignment examples (with sensor values):")
    for ts, dlts, idx, cc in examples:
        row = dl.iloc[idx]
        R(f"\n    aspk_ts={ts}  dl_ts={dlts}  delta={ts-dlts:+d}µs  cc={cc}")
        for col in ["pendulum_state","iteration","target_x","current_x",
                    "velocity","current_angle","angular_velocity"]:
            if col in row:
                R(f"      {col:18s} = {row[col]}")

# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — Ordering structure
# ══════════════════════════════════════════════════════════════════════
def section_5(all_files, func_names):
    R("\n" + "="*80)
    R("# Section 5: Ordering Structure & CFG Validity")
    R("="*80)

    # Load known valid edges
    inv = load_inventory()
    valid_edges = set(zip(inv["func_id"], inv["from_pc"], inv["to_pc"]))
    R(f"\n  Loaded {len(valid_edges)} known valid edges from inventory")

    # 5 example ticks: pick from middle for representative content
    n = len(all_files)
    picks = [all_files[i*n//6] for i in range(1,6)]

    R("\n--- Full ordered controller+math sequences (5 example ticks) ---")
    func_seqs = []
    for cc, ts, p in picks:
        data = p.read_bytes()
        ents = decode_aspk(data)
        ctrl = [(f,fp,tp) for (f,fp,tp) in ents if f in CTRL_MATH]
        func_order = [f for f,_,_ in ctrl]
        func_seqs.append(func_order)
        R(f"\n  cc={cc}  total={len(ents)}  ctrl_math={len(ctrl)}")
        R(f"  Function-id sequence (first 40):")
        seq_str = " ".join(str(f) for f in func_order[:40])
        R(f"    {seq_str}")
        R(f"  Full edge sequence (first 15):")
        for j,(f,fp,tp) in enumerate(ctrl[:15]):
            R(f"    [{j:3d}]  func={f:>3} {func_names.get(f,'?'):25s}  {fp:>6} -> {tp:<6}")

    # Check if function order is consistent across ticks
    R("\n--- Function-order consistency across ticks ---")
    # Compute the "mode" function-id sequence prefix length where all 5 agree
    min_len = min(len(s) for s in func_seqs)
    agree = 0
    for i in range(min_len):
        vals = {s[i] for s in func_seqs}
        if len(vals) == 1:
            agree += 1
        else:
            break
    R(f"  All 5 ticks share the same function-id at positions 0..{agree-1}  (then diverge)")

    # Check sequence-level: is the multiset of (func) the same across all 5?
    multisets = [tuple(sorted(s)) for s in func_seqs]
    same_multiset = len(set(multisets)) == 1
    R(f"  Identical multisets of ctrl+math funcs across 5 ticks: {same_multiset}")

    # Take ALL 100 sample-uniform ticks and compute order entropy at each position
    R("\n--- Position-wise function entropy (across 1000 sampled ticks) ---")
    sample_idxs = np.linspace(0, n-1, 1000, dtype=int)
    samples = [all_files[i] for i in sample_idxs]
    pos_funcs = defaultdict(list)
    seq_lengths = []
    for cc, ts, p in samples:
        data = p.read_bytes()
        ents = decode_aspk(data)
        ctrl = [f for f,_,_ in ents if f in CTRL_MATH]
        seq_lengths.append(len(ctrl))
        for i, f in enumerate(ctrl[:200]):
            pos_funcs[i].append(f)
    R(f"  Sampled {len(samples)} ticks; ctrl+math seq length: median={int(np.median(seq_lengths))}, max={max(seq_lengths)}")
    n_const = sum(1 for i in sorted(pos_funcs)[:200] if len(set(pos_funcs[i]))==1)
    R(f"  Position 0..199: {n_const}/200 positions have a CONSTANT func across all ticks")
    # First 30 positions
    R("  Position-wise entropy (first 30 positions, |unique|/N):")
    line = []
    for i in range(min(30, max(pos_funcs)+1)):
        line.append(f"p{i}={len(set(pos_funcs[i]))}")
    R("    " + "  ".join(line))

    # CFG validity of consecutive pairs
    R("\n--- CFG-validity of consecutive edge pairs ---")
    R("  Testing: for consecutive entries e1=(f1,fp1,tp1), e2=(f2,fp2,tp2),")
    R("    intra-function pairs (f1==f2): is fp2 == tp1? (sequential CFG step)")
    R("    inter-function pairs (f1!=f2): always allowed (call/return)")
    n_pairs = 0
    n_intra = 0
    n_intra_seq = 0
    n_e2_in_inv = 0
    for cc, ts, p in samples[:200]:
        data = p.read_bytes()
        ents = decode_aspk(data)
        for j in range(len(ents)-1):
            e1 = ents[j]; e2 = ents[j+1]
            n_pairs += 1
            if e2 in valid_edges:
                n_e2_in_inv += 1
            if e1[0] == e2[0]:
                n_intra += 1
                if e2[1] == e1[2]:
                    n_intra_seq += 1
    R(f"  Pairs examined: {n_pairs}")
    R(f"  Edges that are in inventory: {n_e2_in_inv}/{n_pairs}  ({100*n_e2_in_inv/n_pairs:.2f}%)")
    R(f"  Intra-function pairs: {n_intra} ({100*n_intra/n_pairs:.2f}%)")
    if n_intra > 0:
        R(f"  Intra-function pairs that are sequential CFG steps (fp2==tp1): "
          f"{n_intra_seq}/{n_intra} ({100*n_intra_seq/n_intra:.2f}%)")

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    R("# ASPK Binary Trace Format EDA")
    R(f"# Session: {SESSION}")
    R(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    R(f"# Format: 8-byte header + N x 8B entries (to_pc:u24, from_pc:u24, func_id:u16) LE")

    func_names = load_func_names()
    R("\nListing aspk files...")
    all_files = list_aspk_sorted()
    R(f"  {len(all_files)} aspk files found")

    section_1(all_files, func_names)
    df, samples = section_2(all_files, func_names)
    section_3(samples, func_names)
    section_4(all_files)
    section_5(all_files, func_names)

    R(f"\n\nTotal runtime: {(time.time()-t0)/60:.1f} min")
    save_report()
    R(f"\nReport written to {REPORT_PATH}")

if __name__ == "__main__":
    main()
