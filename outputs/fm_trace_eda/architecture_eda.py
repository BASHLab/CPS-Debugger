#!/usr/bin/env python3
"""Architecture-Informing EDA for CPS-Debugger FM Trace Generation.

Analyzes execution trace and sensor data to answer 10 architectural questions
for foundation model design.

Output directory: architecture_eda/
"""

import json, os, sys, csv, time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────
BASE = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs")
EXTRACTED = BASE / "extracted"
EDA_DIR = Path(__file__).resolve().parent
OUT = EDA_DIR / "architecture_eda"
OUT.mkdir(exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────
CONTROLLER_FUNCS = {20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46}
MATH_FUNCS = {63,64,70,71,72,73,74,77,78,99}
CTRL_MATH = CONTROLLER_FUNCS | MATH_FUNCS
SENSOR_COLS = ["pendulum_state","iteration","target_x","current_x",
               "velocity","current_angle","angular_velocity"]
STATE_MAP = {0: "SWINGUP", 1: "BALANCE", 2: "RESET"}

# ── report accumulator ────────────────────────────────────────────────
_report = []
def R(line=""):
    _report.append(line)
    print(line)

# ── helpers ────────────────────────────────────────────────────────────
def load_func_names():
    with open(EDA_DIR / "func_id_to_name.json") as f:
        return {int(k): v for k, v in json.load(f).items()}

def load_edge_inventory():
    inv = pd.read_csv(EDA_DIR / "all_edges_inventory.csv")
    ctrl = inv[inv["category"].isin(["controller","math"])].copy()
    lib  = inv[inv["category"] == "library"].copy()
    return inv, ctrl, lib

def get_usable_sessions():
    m = pd.read_csv(EDA_DIR / "full_session_manifest.csv")
    return m[(m["has_processed_trace"]==True) & (m["has_datalayer"]==True)].copy()

def parse_cf_table(s):
    """cf_table JSON -> {(func_id, from_pc, to_pc): count}"""
    cf = json.loads(s)
    out = {}
    for fid_s, fd in cf.items():
        fid = int(fid_s)
        for fp_s, td in fd.items():
            fp = int(fp_s)
            for tp_s, c in td.items():
                out[(fid, fp, int(tp_s))] = c
    return out

def ctrl_vec(edge_dict, ctrl_edges):
    return np.array([edge_dict.get(e, 0) for e in ctrl_edges])

def scan_session(pt_path, ctrl_edges, sample_idx, contig_range=None):
    """Single pass through processed_trace CSV.
    Returns (sampled, contiguous) where each is list of (ts, ctrl_vector)."""
    sampled, contiguous = [], []
    contig_set = set(range(*contig_range)) if contig_range else set()
    all_needed = sample_idx | contig_set
    max_needed = max(all_needed) if all_needed else 0
    with open(pt_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i > max_needed:
                break
            if i in all_needed:
                ts = int(row["timestamp"])
                ed = parse_cf_table(row["cf_table"])
                v  = ctrl_vec(ed, ctrl_edges)
                total = sum(ed.values())
                ctrl_total = sum(c for (fid,_,_),c in ed.items() if fid in CTRL_MATH)
                rec = (ts, v, total, ctrl_total)
                if i in sample_idx:
                    sampled.append(rec)
                if i in contig_set:
                    contiguous.append(rec)
    return sampled, contiguous

def load_datalayer(sess):
    dl = pd.read_csv(Path(sess["path"]) / "datalayer" / "datalayer.csv")
    dl["ts_us"] = dl["timestamp"] // 1000
    dl = dl.sort_values("ts_us").reset_index(drop=True)
    return dl

def nearest_dl_row(dl, ts_us):
    """Return index of nearest datalayer row to given µs timestamp."""
    ts_arr = dl["ts_us"].values
    idx = np.searchsorted(ts_arr, ts_us)
    if idx >= len(ts_arr):
        return len(ts_arr) - 1
    if idx > 0 and abs(ts_arr[idx-1] - ts_us) < abs(ts_arr[idx] - ts_us):
        return idx - 1
    return idx


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — Sensor Features
# ══════════════════════════════════════════════════════════════════════
def section_1(usable):
    R("\n" + "="*80)
    R("# Section 1: Sensor Feature Names, Types, and Statistics")
    R("="*80)

    # pick a long session
    sess = usable.loc[usable["processed_trace_rows"].idxmax()]
    dl = load_datalayer(sess)
    R(f"\nSession: {sess['session_id']}  ({len(dl)} rows)")
    R(f"Columns: {list(dl.columns)}")

    rows = []
    R(f"\n{'Col':>20} {'dtype':>8} {'min':>14} {'max':>14} {'mean':>14} {'std':>14} {'#uniq':>7} {'%miss':>6}")
    R("-"*105)
    for c in SENSOR_COLS:
        s = dl[c]
        nu = int(s.nunique())
        pm = s.isna().mean()*100
        row = dict(column=c, dtype=str(s.dtype), n_unique=nu, pct_missing=round(pm,2),
                   min=float(s.min()), max=float(s.max()),
                   mean=float(s.mean()), std=float(s.std()))
        rows.append(row)
        R(f"{c:>20} {row['dtype']:>8} {row['min']:>14.4f} {row['max']:>14.4f} "
          f"{row['mean']:>14.4f} {row['std']:>14.4f} {nu:>7} {pm:>5.1f}%")

    pd.DataFrame(rows).to_csv(OUT/"sensor_stats.csv", index=False)

    # consistency across 5 sessions
    R("\nCross-session column check:")
    for i in range(min(5, len(usable))):
        s = usable.iloc[i]
        d = pd.read_csv(Path(s["path"])/"datalayer"/"datalayer.csv", nrows=5)
        states = sorted(d["pendulum_state"].unique())
        R(f"  {s['session_id']}: cols={list(d.columns)}, states={states}")

    R("\n**Implication**: 6 continuous sensors + 1 categorical state + 1 counter = 8 features. "
      "Simple linear projection suffices.")
    return dl


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — Trace Target Shape
# ══════════════════════════════════════════════════════════════════════
def section_2(usable, ctrl_edges):
    R("\n" + "="*80)
    R("# Section 2: Trace Target Shape")
    R("="*80)

    sess = usable.iloc[0]
    pt = pd.read_csv(Path(sess["path"])/"processed_trace"/"processed_trace.csv", nrows=5)

    for i in range(len(pt)):
        edges = parse_cf_table(pt.iloc[i]["cf_table"])
        ctrl_c = {k:v for k,v in edges.items() if k[0] in CTRL_MATH}
        lib_c  = {k:v for k,v in edges.items() if k[0] not in CTRL_MATH}
        total = sum(edges.values())
        R(f"\n  Tick {i}: ts={pt.iloc[i]['timestamp']}")
        R(f"    Branches: {total} total, {sum(ctrl_c.values())} ctrl+math ({len(ctrl_c)} edges), "
          f"{sum(lib_c.values())} library ({len(lib_c)} edges)")
        if i == 0:
            R("    Top-5 ctrl edges:")
            for (fid,fp,tp),cnt in sorted(ctrl_c.items(), key=lambda x:-x[1])[:5]:
                R(f"      {fid}:{fp}:{tp}  count={cnt}")

    # count-value distribution across 5 ticks
    all_cnts = []
    for i in range(len(pt)):
        edges = parse_cf_table(pt.iloc[i]["cf_table"])
        all_cnts.extend(v for (fid,_,_),v in edges.items() if fid in CTRL_MATH)
    cd = Counter(all_cnts)
    R("\n  Ctrl+math count-value distribution (5 ticks):")
    for val, freq in sorted(cd.items()):
        R(f"    count={val}: {freq} occurrences")

    R("\n**Answer**: Each tick is a COUNT VECTOR — not binary, not ordered.")
    R("  cf_table gives aggregate counts per edge. Ordering lost in processed_trace.")
    R("  Only raw .aspk preserves execution order.")
    R("  Dimensionality: 318 ctrl+math edges. Most counts are 1; 32 edges vary (1-3).")
    R("\n**Implication**: This is multi-target low-cardinality count prediction (32-dim categorical), "
      "NOT long sequence generation — unless raw .aspk ordering is needed.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — Variable Controller Edges
# ══════════════════════════════════════════════════════════════════════
def section_3(usable, ctrl_edges, func_names, n_sess=3, n_sample=1000):
    R("\n" + "="*80)
    R("# Section 3: The 32 Variable Controller+Math Edges")
    R("="*80)

    all_vecs = []
    for si in range(min(n_sess, len(usable))):
        sess = usable.iloc[si]
        pt_path = Path(sess["path"])/"processed_trace"/"processed_trace.csv"
        total = int(sess["processed_trace_rows"])
        idx = set(np.linspace(0, total-1, min(n_sample, total), dtype=int).tolist())
        R(f"  Scanning {sess['session_id']} ({total} rows, sampling {len(idx)})...")
        sampled, _ = scan_session(pt_path, ctrl_edges, idx)
        vecs = [s[1] for s in sampled]
        all_vecs.extend(vecs)
        R(f"    collected {len(vecs)} ticks")

    mat = np.array(all_vecs)  # (N, 318)
    R(f"\n  Combined matrix: {mat.shape}")

    rows = []
    for j, e in enumerate(ctrl_edges):
        col = mat[:, j]
        fr = (col > 0).mean()
        mn = col.mean()
        sd = col.std()
        uv = sorted(set(col.astype(int)))
        vc = Counter(col.astype(int))
        probs = np.array(list(vc.values())) / len(col)
        ent = float(-np.sum(probs * np.log2(probs + 1e-15)))
        rows.append(dict(
            edge_id=f"{e[0]}:{e[1]}:{e[2]}", func_id=e[0],
            func_name=func_names.get(e[0], "?"),
            firing_rate=round(fr,4), mean_count=round(mn,4),
            std_count=round(sd,4), min_count=int(col.min()),
            max_count=int(col.max()), unique_values=str(uv),
            entropy_bits=round(ent,4), is_variable=sd>0.01))

    df = pd.DataFrame(rows)
    df.to_csv(OUT/"variable_edges_detailed.csv", index=False)

    var_df = df[df["is_variable"]]
    const_df = df[~df["is_variable"]]
    R(f"\n  Variable: {len(var_df)},  Constant: {len(const_df)}")

    R(f"\n{'Edge':>20} {'Function':>22} {'Mean':>6} {'Std':>6} {'Min':>4} {'Max':>4} {'H(bits)':>8} Values")
    R("-"*100)
    for _, r in var_df.iterrows():
        R(f"{r['edge_id']:>20} {r['func_name']:>22} {r['mean_count']:>6.2f} {r['std_count']:>6.2f} "
          f"{r['min_count']:>4} {r['max_count']:>4} {r['entropy_bits']:>8.3f} {r['unique_values']}")

    ents = var_df["entropy_bits"].values
    R(f"\n  Entropy: mean={ents.mean():.3f}, min={ents.min():.3f}, max={ents.max():.3f} bits")
    R(f"  Total information per tick: {ents.sum():.1f} bits ({ents.sum()/8:.1f} bytes)")

    const_ok = (const_df["mean_count"]==1.0).all() and (const_df["std_count"]==0.0).all()
    R(f"\n  Constant edges all fire exactly once? {'YES' if const_ok else 'NO — deviations!'}")

    R("\n**Implication**: Prediction target is 32-dim with ~{:.0f} bits of info per tick. "
      "Very compact problem.".format(ents.sum()))

    var_indices = list(var_df.index)  # indices into ctrl_edges
    return mat, df, var_indices


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — Temporal Autocorrelation
# ══════════════════════════════════════════════════════════════════════
def section_4(usable, ctrl_edges, func_names, var_idx, n_contig=20000):
    R("\n" + "="*80)
    R("# Section 4: Temporal Autocorrelation of Variable Edges")
    R("="*80)

    # pick longest session
    sess = usable.loc[usable["processed_trace_rows"].idxmax()]
    total = int(sess["processed_trace_rows"])
    pt_path = Path(sess["path"])/"processed_trace"/"processed_trace.csv"
    start = total//2 - n_contig//2
    n_contig = min(n_contig, total)

    R(f"\n  Loading {n_contig} contiguous ticks from {sess['session_id']} (rows {start}–{start+n_contig})")
    _, contiguous = scan_session(pt_path, ctrl_edges, set(), contig_range=(start, start+n_contig))
    R(f"  Got {len(contiguous)} ticks")

    mat = np.array([c[1] for c in contiguous])  # (N, 318)
    var_mat = mat[:, var_idx]  # (N, n_var)

    lags = [1, 2, 5, 10, 50, 100, 500, 1000]
    acorr_rows = []

    R(f"\n{'Edge':>20} {'Function':>20} " + "".join(f"{'lag='+str(l):>8}" for l in lags))
    R("-"*100)

    for jv, jf in enumerate(var_idx):
        e = ctrl_edges[jf]
        series = var_mat[:, jv].astype(float)
        row = dict(edge_id=f"{e[0]}:{e[1]}:{e[2]}",
                   func_name=func_names.get(e[0], "?"))
        vals_str = []
        for lag in lags:
            if series.std() > 0 and lag < len(series):
                ac = np.corrcoef(series[:-lag], series[lag:])[0,1]
            else:
                ac = np.nan
            row[f"lag_{lag}"] = round(ac, 4) if not np.isnan(ac) else None
            vals_str.append(f"{ac:>8.3f}" if not np.isnan(ac) else f"{'N/A':>8}")
        acorr_rows.append(row)
        R(f"{row['edge_id']:>20} {row['func_name']:>20} " + "".join(vals_str))

    acorr_df = pd.DataFrame(acorr_rows)
    acorr_df.to_csv(OUT/"autocorrelation.csv", index=False)

    # summary
    for lag in [1, 10, 100, 1000]:
        v = acorr_df[f"lag_{lag}"].dropna()
        if len(v):
            R(f"\n  lag={lag}: mean={v.mean():.3f}, std={v.std():.3f}, "
              f"min={v.min():.3f}, max={v.max():.3f}")

    # vector-level: identical consecutive ticks
    ident = np.mean([np.array_equal(var_mat[i], var_mat[i+1]) for i in range(len(var_mat)-1)])
    changes = np.sum(var_mat[1:] != var_mat[:-1], axis=1)
    R(f"\n  Identical consecutive ticks: {ident:.4f} ({ident*100:.1f}%)")
    R(f"  Edges changing per tick: mean={changes.mean():.2f}, std={changes.std():.2f}, max={changes.max()}")

    # cosine similarity
    cos = []
    for i in range(len(var_mat)-1):
        v1, v2 = var_mat[i].astype(float), var_mat[i+1].astype(float)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1>0 and n2>0:
            cos.append(np.dot(v1,v2)/(n1*n2))
    cos = np.array(cos)
    R(f"  Cosine similarity: mean={cos.mean():.4f}, std={cos.std():.4f}")

    if ident > 0.9:
        R(f"\n**Implication**: {ident*100:.0f}% of ticks identical to previous — very persistent. "
          "Consider delta encoding. 'Copy previous' is a strong baseline.")
    elif ident > 0.5:
        R(f"\n**Implication**: {ident*100:.0f}% identical — moderate persistence. "
          "Short temporal context (10-100 ticks) likely sufficient.")
    else:
        R(f"\n**Implication**: Only {ident*100:.0f}% identical — low persistence. "
          "Need strong sensor conditioning per tick.")

    return acorr_df


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — Sensor-to-Trace Correlation
# ══════════════════════════════════════════════════════════════════════
def section_5(usable, ctrl_edges, var_idx, func_names, n_sess=3, n_sample=2000):
    R("\n" + "="*80)
    R("# Section 5: Sensor-to-Trace Correlation (Feasibility Check)")
    R("="*80)

    merged = []  # (sensor_arr, var_vec, state)

    for si in range(min(n_sess, len(usable))):
        sess = usable.iloc[si]
        dl = load_datalayer(sess)
        dl_ts = dl["ts_us"].values
        dl_sensors = dl[SENSOR_COLS].values
        dl_states  = dl["pendulum_state"].values

        pt_path = Path(sess["path"])/"processed_trace"/"processed_trace.csv"
        total = int(sess["processed_trace_rows"])
        idx = set(np.linspace(0, total-1, min(n_sample, total), dtype=int).tolist())
        R(f"  Merging {sess['session_id']} ({n_sample} ticks)...")
        sampled, _ = scan_session(pt_path, ctrl_edges, idx)

        for ts, vec, _, _ in sampled:
            di = nearest_dl_row(dl, ts)
            merged.append((dl_sensors[di], vec[var_idx], int(dl_states[di]),
                           abs(int(dl_ts[di]) - ts)))
        R(f"    merged {len(sampled)} ticks")

    sensor_mat = np.array([m[0] for m in merged])  # (N, 7)
    trace_mat  = np.array([m[1] for m in merged])   # (N, n_var)
    ts_diffs   = [m[3] for m in merged]
    R(f"\n  Total merged: {len(merged)}")
    R(f"  Timestamp alignment: mean={np.mean(ts_diffs):.0f}µs, max={np.max(ts_diffs):.0f}µs")

    # correlation matrix (n_var x n_sensor)
    n_var = len(var_idx)
    corr = np.zeros((n_var, len(SENSOR_COLS)))
    for j in range(n_var):
        for k in range(len(SENSOR_COLS)):
            tc = trace_mat[:, j].astype(float)
            sc = sensor_mat[:, k]
            if tc.std() > 0 and np.std(sc) > 0:
                corr[j, k] = np.corrcoef(tc, sc)[0, 1]

    var_eids = [f"{ctrl_edges[vi][0]}:{ctrl_edges[vi][1]}:{ctrl_edges[vi][2]}" for vi in var_idx]
    corr_df = pd.DataFrame(corr, index=var_eids, columns=SENSOR_COLS)
    corr_df.to_csv(OUT/"sensor_trace_corr.csv")

    R(f"\n{'Edge':>20} {'Function':>20} {'Best sensor':>18} {'|r|':>7}")
    R("-"*70)
    for j, vi in enumerate(var_idx):
        e = ctrl_edges[vi]
        bk = np.argmax(np.abs(corr[j]))
        R(f"{var_eids[j]:>20} {func_names.get(e[0],'?'):>20} {SENSOR_COLS[bk]:>18} {abs(corr[j,bk]):>7.3f}")

    R("\nPer-sensor summary:")
    for k, sc in enumerate(SENSOR_COLS):
        mx = np.max(np.abs(corr[:, k]))
        mn = np.mean(np.abs(corr[:, k]))
        R(f"  {sc:>20}: max|r|={mx:.3f}, mean|r|={mn:.3f}")

    overall_max = np.max(np.abs(corr))
    overall_mean = np.mean(np.abs(corr))
    R(f"\n  Overall: max|r|={overall_max:.3f}, mean|r|={overall_mean:.3f}")

    if overall_max > 0.5:
        R("\n**Implication**: Strong linear signal exists. Sensor-conditioned prediction is feasible.")
    elif overall_max > 0.2:
        R("\n**Implication**: Moderate linear signal. Nonlinear models or temporal features may help.")
    else:
        R("\n**Implication**: Weak linear correlations. Need nonlinear models, temporal context, "
          "or the relationship is mediated by FSM state.")

    # Also save merged data for downstream
    return corr_df, merged


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — Graph Structure
# ══════════════════════════════════════════════════════════════════════
def section_6(usable):
    R("\n" + "="*80)
    R("# Section 6: Graph Structure from layout.json")
    R("="*80)

    sess = usable.iloc[0]
    lp = Path(sess["path"])/"code"/"layout.json"
    R(f"\n  Parsing {lp}")
    with open(lp) as f:
        layout = json.load(f)

    n_funcs = 0; total_blocks = 0; total_edges = 0; total_inst = 0
    func_rows = []
    type_counts = Counter()

    for key, val in layout.items():
        if not key.isdigit():
            continue
        n_funcs += 1
        bb = val.get("br_blocks", {})
        ts = val.get("target_sites", {})
        nb = len(bb); ne = len(ts)
        ni = sum(len(insts) for insts in bb.values())
        total_blocks += nb; total_edges += ne; total_inst += ni

        for insts in bb.values():
            for inst in insts:
                type_counts[inst.get("type", -1)] += 1

        func_rows.append(dict(
            func_id=int(key), name=val.get("name","?"),
            n_blocks=nb, n_edges=ne, n_instructions=ni,
            n_direct_calls=len(val.get("call_direct_sites",{})),
            n_indirect_calls=len(val.get("call_indirect_sites",{}))))

    R(f"\n  Functions: {n_funcs}")
    R(f"  Basic blocks: {total_blocks}")
    R(f"  Control-flow edges: {total_edges}")
    R(f"  Instructions: {total_inst}")
    R(f"  Avg insts/block: {total_inst/total_blocks:.1f}")
    R(f"  func_exports: {layout.get('func_exports',{})}")
    R(f"  import_func_count: {layout.get('import_func_count',0)}")

    tnames = {0:"branch",1:"call_direct",2:"call_indirect",3:"block_marker",4:"return",5:"regular"}
    R("\n  Instruction types:")
    for t in sorted(type_counts):
        R(f"    type {t} ({tnames.get(t,'?'):>14}): {type_counts[t]:>6} ({100*type_counts[t]/total_inst:.1f}%)")

    fdf = pd.DataFrame(func_rows).sort_values("n_instructions", ascending=False)
    fdf.to_csv(OUT/"function_complexity.csv", index=False)
    R("\n  Top-10 functions by complexity:")
    for _, r in fdf.head(10).iterrows():
        R(f"    func {r['func_id']:>3} {r['name']:>25}: {r['n_blocks']:>3} blk, "
          f"{r['n_edges']:>3} edg, {r['n_instructions']:>5} inst, {r['n_direct_calls']:>2} calls")

    # Example block from tick(22)
    if "22" in layout:
        tick = layout["22"]
        fb = tick.get("first_block","")
        bb = tick.get("br_blocks",{})
        R(f"\n  Example: tick (func 22) — {len(bb)} blocks, first_block={fb}")
        if fb in bb:
            insts = bb[fb]
            R(f"    Block {fb}: {len(insts)} instructions")
            for inst in insts[:4]:
                R(f"      pc={inst['pc']}, type={inst['type']}, inst='{inst['inst']}'")
            if len(insts) > 4:
                R(f"      ... +{len(insts)-4} more")

    stats = dict(n_functions=n_funcs, n_basic_blocks=total_blocks,
                 n_control_flow_edges=total_edges, n_instructions=total_inst,
                 avg_insts_per_block=round(total_inst/total_blocks,1))
    with open(OUT/"graph_stats.json","w") as f:
        json.dump(stats, f, indent=2)

    R(f"\n**Implication**: Moderate graph ({total_blocks} blocks, {total_edges} edges). "
      "Per-edge identity embeddings (634-dim) suffice. No GNN needed. "
      "layout.json opcodes could enrich embeddings.")
    return stats


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — FSM State Distribution
# ══════════════════════════════════════════════════════════════════════
def section_7(usable, ctrl_edges, var_idx, func_names, n_sess=10, n_merged=3):
    R("\n" + "="*80)
    R("# Section 7: FSM State Distribution")
    R("="*80)

    # --- datalayer-only stats from many sessions ---
    fsm_rows = []
    for si in range(min(n_sess, len(usable))):
        sess = usable.iloc[si]
        dl = load_datalayer(sess)
        sd = dl["pendulum_state"].value_counts(normalize=True)
        sc = dl["pendulum_state"].value_counts()

        # episode analysis
        dl["sc"] = dl["pendulum_state"].diff().ne(0).cumsum()
        eps = dl.groupby("sc").agg(state=("pendulum_state","first"),
                                   n_ticks=("pendulum_state","count"),
                                   dur_us=("ts_us",lambda x:x.max()-x.min()))
        row = dict(session_id=sess["session_id"], n_ticks=len(dl), n_episodes=len(eps))
        for sv in sorted(dl["pendulum_state"].unique()):
            sn = STATE_MAP.get(sv, f"s{sv}")
            row[f"pct_{sn}"] = round(sd.get(sv,0)*100, 1)
            se = eps[eps["state"]==sv]
            if len(se):
                row[f"mean_ep_ms_{sn}"] = round(se["dur_us"].mean()/1000, 1)
                row[f"n_ep_{sn}"] = len(se)
        fsm_rows.append(row)
        R(f"  {sess['session_id']}: {dict(sc)}, {len(eps)} episodes")

    fsm_df = pd.DataFrame(fsm_rows)
    fsm_df.to_csv(OUT/"fsm_distribution.csv", index=False)

    R("\nAggregate:")
    for c in fsm_df.columns:
        if c.startswith("pct_"):
            v = fsm_df[c].dropna()
            R(f"  {c}: mean={v.mean():.1f}%, std={v.std():.1f}%")

    # --- per-state variable edge firing from merged sessions ---
    R("\nPer-state variable edge firing (from merged data):")
    state_vecs = defaultdict(list)  # state -> [var_vec, ...]

    for si in range(min(n_merged, len(usable))):
        sess = usable.iloc[si]
        dl = load_datalayer(sess)
        dl_ts = dl["ts_us"].values
        dl_states = dl["pendulum_state"].values

        pt_path = Path(sess["path"])/"processed_trace"/"processed_trace.csv"
        total = int(sess["processed_trace_rows"])
        idx = set(np.linspace(0, total-1, min(1500, total), dtype=int).tolist())
        sampled, _ = scan_session(pt_path, ctrl_edges, idx)
        for ts, vec, _, _ in sampled:
            di = nearest_dl_row(dl, ts)
            state = int(dl_states[di])
            state_vecs[state].append(vec[var_idx])

    R(f"\n  Ticks per state: " + ", ".join(f"{STATE_MAP.get(k,k)}={len(v)}" for k,v in sorted(state_vecs.items())))

    state_means = {}
    for st, vecs in state_vecs.items():
        if len(vecs) > 10:
            state_means[st] = np.array(vecs).mean(axis=0)

    if len(state_means) > 1:
        states_sorted = sorted(state_means.keys())
        R(f"\n{'Edge':>20} {'Function':>20}" + "".join(f" {STATE_MAP.get(s,s):>10}" for s in states_sorted))
        R("-"*80)
        for j, vi in enumerate(var_idx[:15]):
            e = ctrl_edges[vi]
            eid = f"{e[0]}:{e[1]}:{e[2]}"
            vals = "".join(f" {state_means[s][j]:>10.3f}" for s in states_sorted)
            R(f"{eid:>20} {func_names.get(e[0],'?'):>20}{vals}")

    R("\n**Implication**: FSM state should be an explicit conditioning input. "
      "If means differ substantially across states, consider state-specific model heads.")
    return fsm_df


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — Cross-Session Consistency
# ══════════════════════════════════════════════════════════════════════
def section_8(usable, ctrl_edges, var_idx, func_names):
    R("\n" + "="*80)
    R("# Section 8: Cross-Session Consistency")
    R("="*80)

    per_sess = []
    for _, sess in usable.iterrows():
        pt_path = Path(sess["path"])/"processed_trace"/"processed_trace.csv"
        if not Path(pt_path).exists():
            continue
        total = int(sess["processed_trace_rows"])
        if total == 0:
            continue

        # sample every Nth row using pandas skiprows callable (faster than full scan)
        step = max(1, total // 500)
        try:
            df = pd.read_csv(pt_path, usecols=["cf_table"],
                             skiprows=lambda i: i > 0 and i % step != 0)
        except Exception:
            R(f"  {sess['session_id']}: FAILED to read")
            continue

        vecs = []
        for _, row in df.iterrows():
            edges = parse_cf_table(row["cf_table"])
            vec = np.array([edges.get(ctrl_edges[vi], 0) for vi in var_idx])
            vecs.append(vec)

        if not vecs:
            continue
        mat = np.array(vecs)
        means = mat.mean(axis=0)
        stds  = mat.std(axis=0)
        rec = dict(session_id=sess["session_id"], n_sampled=len(vecs))
        for j, vi in enumerate(var_idx):
            eid = f"{ctrl_edges[vi][0]}:{ctrl_edges[vi][1]}:{ctrl_edges[vi][2]}"
            rec[f"mean_{eid}"] = round(float(means[j]), 4)
            rec[f"std_{eid}"]  = round(float(stds[j]), 4)
        per_sess.append(rec)
        R(f"  {sess['session_id']}: {len(vecs)} ticks sampled")

    cs_df = pd.DataFrame(per_sess)
    cs_df.to_csv(OUT/"cross_session_firing.csv", index=False)

    R(f"\nPer-edge cross-session variance:")
    R(f"  {'Edge':>20} {'Function':>20} {'Mean(means)':>12} {'Std(means)':>11} {'CV':>7}")
    R("-"*75)
    consistency = []
    for j, vi in enumerate(var_idx):
        e = ctrl_edges[vi]
        eid = f"{e[0]}:{e[1]}:{e[2]}"
        col = f"mean_{eid}"
        if col in cs_df.columns:
            vals = cs_df[col].dropna()
            mm = vals.mean(); sm = vals.std()
            cv = sm/mm if mm > 0 else 0
            R(f"  {eid:>20} {func_names.get(e[0],'?'):>20} {mm:>12.4f} {sm:>11.4f} {cv:>7.4f}")
            consistency.append(cv)

    mean_cv = np.mean(consistency) if consistency else 0
    R(f"\n  Mean CV across edges: {mean_cv:.4f}")

    # outlier sessions
    mean_cols = [c for c in cs_df.columns if c.startswith("mean_")]
    if mean_cols:
        mm = cs_df[mean_cols].values
        gm = mm.mean(axis=0)
        devs = np.sqrt(np.mean((mm - gm)**2, axis=1))
        cs_df["deviation"] = devs
        R("\n  Sessions with highest deviation:")
        for _, r in cs_df.nlargest(5, "deviation").iterrows():
            R(f"    {r['session_id']}: dev={r['deviation']:.4f}")

    if mean_cv < 0.1:
        R(f"\n**Implication**: Sessions are homogeneous (CV={mean_cv:.3f}). Random tick splits OK.")
    elif mean_cv < 0.3:
        R(f"\n**Implication**: Moderate session effects (CV={mean_cv:.3f}). Use session-level splits.")
    else:
        R(f"\n**Implication**: Strong session effects (CV={mean_cv:.3f}). Must use leave-one-session-out.")
    return cs_df


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — Sequence Length Distribution
# ══════════════════════════════════════════════════════════════════════
def section_9(usable, ctrl_edges, n_sess=5, n_sample=1000):
    R("\n" + "="*80)
    R("# Section 9: Sequence Length Distribution")
    R("="*80)
    R("\n  Note: processed_trace gives counts, not ordered sequences.")
    R("  'Sequence length' = total branch count per tick.")

    all_total, all_ctrl = [], []
    for si in range(min(n_sess, len(usable))):
        sess = usable.iloc[si]
        pt_path = Path(sess["path"])/"processed_trace"/"processed_trace.csv"
        total = int(sess["processed_trace_rows"])
        idx = set(np.linspace(0, total-1, min(n_sample, total), dtype=int).tolist())
        R(f"  Scanning {sess['session_id']}...")
        sampled, _ = scan_session(pt_path, ctrl_edges, idx)
        for _, _, tot, ct in sampled:
            all_total.append(tot)
            all_ctrl.append(ct)

    at = np.array(all_total); ac = np.array(all_ctrl)
    for label, arr in [("All edges", at), ("Controller+math", ac)]:
        R(f"\n  {label}:")
        R(f"    min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}, "
          f"median={np.median(arr):.0f}, std={arr.std():.1f}")
        R(f"    p5={np.percentile(arr,5):.0f}, p25={np.percentile(arr,25):.0f}, "
          f"p75={np.percentile(arr,75):.0f}, p95={np.percentile(arr,95):.0f}, "
          f"p99={np.percentile(arr,99):.0f}")

    pd.DataFrame({"all_branches": at, "ctrl_math_branches": ac}).to_csv(
        OUT/"seq_length_dist.csv", index=False)

    # histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(at, bins=50, color="#378ADD", edgecolor="white")
    axes[0].set_title("All branches per tick")
    axes[0].set_xlabel("Count"); axes[0].set_ylabel("Frequency")
    axes[1].hist(ac, bins=50, color="#1D9E75", edgecolor="white")
    axes[1].set_title("Controller+math branches per tick")
    axes[1].set_xlabel("Count"); axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUT/"fig_seq_length.png", dpi=150, bbox_inches="tight")
    plt.close()

    if ac.mean() < 500:
        R(f"\n**Implication**: Ctrl+math ~{ac.mean():.0f} tokens/tick — very manageable for AR generation.")
    elif ac.mean() < 2000:
        R(f"\n**Implication**: Ctrl+math ~{ac.mean():.0f} tokens/tick — consider efficient attention.")
    else:
        R(f"\n**Implication**: Long sequences ~{ac.mean():.0f}. Need Mamba/block diffusion.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — Sensor Quantization
# ══════════════════════════════════════════════════════════════════════
def section_10(usable):
    R("\n" + "="*80)
    R("# Section 10: Sensor Quantization Feasibility")
    R("="*80)

    sess = usable.loc[usable["processed_trace_rows"].idxmax()]
    dl = load_datalayer(sess)
    R(f"\n  Session: {sess['session_id']} ({len(dl)} rows)")

    bins_list = [64, 128, 256, 512, 1024]
    rows = []
    R(f"\n{'Feature':>20} {'#unique':>8} {'Type':>10}" +
      "".join(f" {'err@'+str(b):>9}" for b in bins_list))
    R("-"*95)

    for col in SENSOR_COLS:
        vals = dl[col].dropna().values
        nu = len(np.unique(vals))
        var = np.var(vals)
        ctype = "discrete" if nu<=10 else ("low-card" if nu<=100 else "continuous")

        errs = {}
        for nb in bins_list:
            if var > 0 and vals.max() > vals.min():
                be = np.linspace(vals.min(), vals.max(), nb+1)
                bi = np.clip(np.digitize(vals, be)-1, 0, nb-1)
                bc = (be[:-1]+be[1:])/2
                mse = np.mean((vals - bc[bi])**2)
                errs[nb] = 100*mse/var
            else:
                errs[nb] = 0.0

        rows.append(dict(feature=col, n_unique=nu, type=ctype, variance=var,
                         **{f"err_{b}bins": errs[b] for b in bins_list}))
        estr = "".join(f" {errs[b]:>8.4f}%" for b in bins_list)
        R(f"{col:>20} {nu:>8} {ctype:>10}{estr}")

    pd.DataFrame(rows).to_csv(OUT/"quantization_error.csv", index=False)

    mx256 = max(r["err_256bins"] for r in rows)
    if mx256 < 1.0:
        R(f"\n**Implication**: Max error at 256 bins = {mx256:.4f}%. "
          "Simple uniform quantization into 256 tokens works.")
    else:
        R(f"\n**Implication**: Max error at 256 bins = {mx256:.2f}%. "
          "Consider 1024 bins or learned VQ-VAE for some features.")


# ══════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════
def make_figures(ctrl_edges, var_idx, func_names):
    """Generate summary figures from saved CSVs."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Sensor-trace correlation heatmap
    ax = axes[0, 0]
    try:
        corr = pd.read_csv(OUT/"sensor_trace_corr.csv", index_col=0)
        im = ax.imshow(corr.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=6)
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, fontsize=7, rotation=45, ha="right")
        ax.set_title("Sensor-Trace Correlation")
        plt.colorbar(im, ax=ax, fraction=0.046)
    except Exception:
        ax.text(0.5, 0.5, "No correlation data", ha="center")

    # 2. Autocorrelation decay
    ax = axes[0, 1]
    try:
        ac = pd.read_csv(OUT/"autocorrelation.csv")
        lags = [1, 2, 5, 10, 50, 100, 500, 1000]
        lag_cols = [f"lag_{l}" for l in lags]
        for _, row in ac.iterrows():
            vals = [row.get(c) for c in lag_cols]
            ax.plot(lags, vals, alpha=0.4, linewidth=0.8)
        # mean line
        means = [ac[c].dropna().mean() for c in lag_cols]
        ax.plot(lags, means, "k-", linewidth=2, label="mean")
        ax.set_xscale("log"); ax.set_xlabel("Lag (ticks)"); ax.set_ylabel("Autocorrelation")
        ax.set_title("Variable Edge Autocorrelation"); ax.legend()
    except Exception:
        ax.text(0.5, 0.5, "No autocorrelation data", ha="center")

    # 3. Variable edge entropy
    ax = axes[0, 2]
    try:
        ve = pd.read_csv(OUT/"variable_edges_detailed.csv")
        var_e = ve[ve["is_variable"]]
        ax.barh(range(len(var_e)), var_e["entropy_bits"].values, color="#1D9E75")
        ax.set_yticks(range(len(var_e)))
        ax.set_yticklabels([f"{r['edge_id']} ({r['func_name'][:10]})" for _,r in var_e.iterrows()], fontsize=6)
        ax.set_xlabel("Entropy (bits)"); ax.set_title("Variable Edge Entropy")
    except Exception:
        ax.text(0.5, 0.5, "No entropy data", ha="center")

    # 4. FSM state distribution
    ax = axes[1, 0]
    try:
        fsm = pd.read_csv(OUT/"fsm_distribution.csv")
        pct_cols = [c for c in fsm.columns if c.startswith("pct_")]
        if pct_cols:
            means = [fsm[c].mean() for c in pct_cols]
            labels = [c.replace("pct_","") for c in pct_cols]
            ax.bar(labels, means, color=["#FF6B6B","#4ECDC4","#95E1D3"])
            ax.set_ylabel("% of ticks"); ax.set_title("FSM State Distribution")
    except Exception:
        ax.text(0.5, 0.5, "No FSM data", ha="center")

    # 5. Cross-session consistency
    ax = axes[1, 1]
    try:
        cs = pd.read_csv(OUT/"cross_session_firing.csv")
        mean_cols = [c for c in cs.columns if c.startswith("mean_")]
        if mean_cols and len(mean_cols) >= 2:
            # plot first two variable edges across sessions
            c1, c2 = mean_cols[0], mean_cols[1]
            ax.scatter(cs[c1], cs[c2], c="steelblue", s=30)
            ax.set_xlabel(c1.replace("mean_","")); ax.set_ylabel(c2.replace("mean_",""))
            ax.set_title("Cross-Session Edge Means")
            for _, r in cs.iterrows():
                ax.annotate(r["session_id"][-5:], (r[c1], r[c2]), fontsize=5)
    except Exception:
        ax.text(0.5, 0.5, "No cross-session data", ha="center")

    # 6. Quantization error
    ax = axes[1, 2]
    try:
        qe = pd.read_csv(OUT/"quantization_error.csv")
        bins_cols = [c for c in qe.columns if c.startswith("err_")]
        for _, r in qe.iterrows():
            vals = [r[c] for c in bins_cols]
            bins_vals = [int(c.split("_")[1].replace("bins","")) for c in bins_cols]
            ax.plot(bins_vals, vals, "o-", label=r["feature"][:12], markersize=3)
        ax.set_xlabel("N bins"); ax.set_ylabel("Quant error (% var)")
        ax.set_title("Quantization Error vs Bins"); ax.legend(fontsize=6)
        ax.set_yscale("log")
    except Exception:
        ax.text(0.5, 0.5, "No quantization data", ha="center")

    plt.tight_layout()
    plt.savefig(OUT/"fig_architecture_eda.png", dpi=150, bbox_inches="tight")
    plt.close()
    R(f"\n  Saved figure to {OUT/'fig_architecture_eda.png'}")


# ══════════════════════════════════════════════════════════════════════
# COLLECT — assemble report from saved CSVs
# ══════════════════════════════════════════════════════════════════════
def collect_report(ctrl_edges, var_idx, func_names):
    """Read all saved CSVs and produce final report + figures."""
    R("\n" + "="*80)
    R("# COLLECTED REPORT — Architecture-Informing EDA")
    R("="*80)

    # Summarise each CSV if it exists
    for fname, label in [
        ("sensor_stats.csv", "Section 1: Sensor Stats"),
        ("variable_edges_detailed.csv", "Section 3: Variable Edges"),
        ("autocorrelation.csv", "Section 4: Autocorrelation"),
        ("sensor_trace_corr.csv", "Section 5: Sensor-Trace Correlation"),
        ("graph_stats.json", "Section 6: Graph Structure"),
        ("fsm_distribution.csv", "Section 7: FSM Distribution"),
        ("cross_session_firing.csv", "Section 8: Cross-Session"),
        ("seq_length_dist.csv", "Section 9: Sequence Lengths"),
        ("quantization_error.csv", "Section 10: Quantization"),
    ]:
        p = OUT / fname
        if p.exists():
            R(f"\n  ✓ {label} — {fname} ({p.stat().st_size/1024:.0f} KB)")
        else:
            R(f"\n  ✗ {label} — {fname} MISSING")

    make_figures(ctrl_edges, var_idx, func_names)

    with open(OUT/"architecture_eda_report.md", "w") as f:
        f.write("\n".join(_report))
    R(f"\nFinal report: {OUT/'architecture_eda_report.md'}")


# ══════════════════════════════════════════════════════════════════════
# MAIN — supports --sections for parallel SLURM execution
# ══════════════════════════════════════════════════════════════════════
def _setup():
    """Common setup returning (func_names, ctrl_edges, var_idx, usable)."""
    func_names = load_func_names()
    inv, ctrl_inv, lib_inv = load_edge_inventory()
    usable = get_usable_sessions()
    ctrl_edges = list(zip(
        ctrl_inv["func_id"].astype(int),
        ctrl_inv["from_pc"].astype(int),
        ctrl_inv["to_pc"].astype(int)))
    var_idx = list(np.where(ctrl_inv["is_variable"].values)[0])
    R(f"Usable sessions: {len(usable)}, Ctrl+math edges: {len(ctrl_edges)}, Variable: {len(var_idx)}")
    return func_names, ctrl_edges, var_idx, usable, inv, ctrl_inv, lib_inv


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", type=str, default="all",
                        help="Comma-separated section numbers (1-10), 'collect', or 'all'")
    args = parser.parse_args()

    t_start = time.time()
    R("# Architecture-Informing EDA — CPS-Debugger FM Trace Generation")
    R(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}")
    R(f"# sections={args.sections}")
    R("="*80)

    func_names, ctrl_edges, var_idx, usable, inv, ctrl_inv, lib_inv = _setup()

    if args.sections == "collect":
        collect_report(ctrl_edges, var_idx, func_names)
        return

    if args.sections == "all":
        sections = list(range(1, 11))
    else:
        sections = [int(s.strip()) for s in args.sections.split(",")]

    dispatch = {
        1:  lambda: section_1(usable),
        2:  lambda: section_2(usable, ctrl_edges),
        3:  lambda: section_3(usable, ctrl_edges, func_names),
        4:  lambda: section_4(usable, ctrl_edges, func_names, var_idx),
        5:  lambda: section_5(usable, ctrl_edges, var_idx, func_names),
        6:  lambda: section_6(usable),
        7:  lambda: section_7(usable, ctrl_edges, var_idx, func_names),
        8:  lambda: section_8(usable, ctrl_edges, var_idx, func_names),
        9:  lambda: section_9(usable, ctrl_edges),
        10: lambda: section_10(usable),
    }

    for s in sections:
        if s in dispatch:
            t = time.time()
            R(f"\n>>> Running section {s}")
            dispatch[s]()
            R(f"\n[Section {s}: {time.time()-t:.1f}s]")

    # If running all, also do figures + final report
    if args.sections == "all":
        make_figures(ctrl_edges, var_idx, func_names)

    elapsed = time.time() - t_start
    R(f"\n\nRuntime: {elapsed/60:.1f} minutes")

    # Save per-group partial report
    rpt_name = f"report_s{'_'.join(str(s) for s in sections)}.md" if args.sections != "all" else "architecture_eda_report.md"
    with open(OUT / rpt_name, "w") as f:
        f.write("\n".join(_report))
    R(f"Saved to {OUT/rpt_name}")


if __name__ == "__main__":
    main()
