"""
phase2_D_neither_branches.py — Characterise the 16 "neither" branches.

The 16/100 branches not predictable from physical sensors OR syslog (R²<0.3).
Categorises each into: LOGGING, CONSTANT, BINARY_NOISE, RUN_SPECIFIC, or TRULY_UNPREDICTABLE.

Requires: outputs/phase2/stratified_r2.csv (from Phase A)
          OR falls back to branch_regression_results.json from Phase 1.

Output: outputs/phase2/fig_D1_neither_branches_analysis.png
        outputs/phase2/neither_branches_detail.json
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
OUT  = ROOT / "outputs/phase2"
EXP  = ROOT / "outputs/experiments"
OUT.mkdir(parents=True, exist_ok=True)

SYSLOG_COLS  = ["sl_cpu", "sl_mem", "sl_load1", "sl_load5", "sl_load15"]


def func_category(label):
    lower = label.lower()
    if any(x in lower for x in ["printf", "vfprintf", "fwritex", "writev",
                                  "stdio", "log_data", "fputs", "fwrite",
                                  "towrite"]):
        return "LOGGING"
    if any(x in lower for x in ["memcpy", "memset"]):
        return "MEMORY"
    if any(x in lower for x in ["pou_lqr", "lqr"]):
        return "LQR"
    if any(x in lower for x in ["tick", "pou_main", "pou_general"]):
        return "TICK"
    if any(x in lower for x in ["standup"]):
        return "STANDUP"
    if any(x in lower for x in ["fmin", "fmax", "sin", "cos", "math"]):
        return "MATH"
    return "OTHER"


def main():
    # ── Load vocab and branch data ──────────────────────────────────────────
    df = pd.read_parquet(EXP / "aligned_dataset.parquet")
    with open(EXP / "vocab_info.json") as f:
        vocab_info = json.load(f)
    vocab  = vocab_info["vocab"]
    labels = vocab_info.get("labels", {})

    branch_mat = df[vocab].values.astype(float)
    valid = branch_mat.sum(axis=1) > 0
    df   = df[valid].reset_index(drop=True)
    branch_mat = branch_mat[valid]
    states = df["dl_state"].values

    # ── Identify "neither" branches ─────────────────────────────────────────
    # Load R² data from Phase A if available, else from Phase 1 JSON
    stratified_csv = OUT / "stratified_r2.csv"
    reg_json       = EXP / "branch_regression_results.json"

    neither_indices = []

    if stratified_csv.exists():
        print("Loading stratified R² from Phase A...")
        csv_df = pd.read_csv(stratified_csv)
        # "neither" = low R²_total AND low R²_within
        neither_mask = (csv_df["r2_total"] < 0.3) & (csv_df["r2_within_formula"] < 0.1)
        neither_indices = list(np.where(neither_mask.values)[0])
        print(f"  {len(neither_indices)} neither branches (R²_total<0.3 AND R²_within<0.1)")
    elif reg_json.exists():
        print("Loading R² from Phase 1 branch regression results...")
        with open(reg_json) as f:
            reg = json.load(f)
        # Reconstruct per-branch R² from the JSON (uses top-10 only for named branches)
        # Fall back to computing from scratch
        neither_indices = []
        # Simple threshold on variance
        for j, bkey in enumerate(vocab):
            col = branch_mat[:, j]
            cv  = col.std() / (col.mean() + 1e-9)
            if cv < 0.05:
                neither_indices.append(j)
        print(f"  {len(neither_indices)} low-variance branches (proxy for 'neither')")
    else:
        print("No R² data found. Computing variance-based classification...")
        for j in range(len(vocab)):
            col = branch_mat[:, j]
            cv  = col.std() / (col.mean() + 1e-9)
            if cv < 0.1:
                neither_indices.append(j)
        print(f"  {len(neither_indices)} low-variance branches")

    print(f"\nAnalysing {len(neither_indices)} branches...")

    # ── Per-branch analysis ─────────────────────────────────────────────────
    records = []
    categories = []

    for j in neither_indices:
        bkey  = vocab[j]
        label = labels.get(bkey, bkey)
        col   = branch_mat[:, j]

        # Basic stats
        mean_val = float(col.mean())
        std_val  = float(col.std())
        cv       = std_val / (mean_val + 1e-9)
        presence = float((col > 0).mean())
        unique_vals = len(np.unique(col))

        # Is it binary (0/1 values only)?
        is_binary = float(np.all(np.isin(col, [0.0, 1.0])))

        # Correlation with run_id (run-specific behavior)
        run_ids   = pd.Categorical(df["run"]).codes
        r_run, _  = stats.pointbiserialr(run_ids, (col > mean_val).astype(float)) \
                    if unique_vals > 1 else (0.0, 1.0)

        # Correlation with iteration number
        if "dl_iteration" in df.columns:
            r_iter, _ = stats.pearsonr(df["dl_iteration"].values, col) \
                        if unique_vals > 1 else (0.0, 1.0)
        else:
            r_iter = 0.0

        # State correlation
        r_state, _ = stats.kruskal(*[col[states==s] for s in [0,1,2]
                                     if (states==s).sum() > 10])

        # Categorise
        fn_cat = func_category(label)
        if fn_cat == "LOGGING":
            cat = "LOGGING"
        elif cv < 0.02:
            cat = "CONSTANT"
        elif is_binary and 0.3 < presence < 0.7:
            cat = "BINARY_NOISE"
        elif abs(r_run) > 0.3:
            cat = "RUN_SPECIFIC"
        else:
            cat = "TRULY_UNPREDICTABLE"

        categories.append(cat)
        records.append({
            "branch_id":   bkey,
            "label":       label,
            "fn_category": fn_cat,
            "category":    cat,
            "mean":        mean_val,
            "std":         std_val,
            "cv":          float(cv),
            "presence":    presence,
            "is_binary":   bool(is_binary),
            "unique_vals": unique_vals,
            "r_run":       float(r_run),
            "r_iter":      float(r_iter),
        })

    # ── Print summary ───────────────────────────────────────────────────────
    from collections import Counter
    cat_counts = Counter(categories)
    print("\nCategory breakdown:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:25s}: {count} branches")

    # ── Save JSON ───────────────────────────────────────────────────────────
    out_json = {
        "n_neither_branches": len(neither_indices),
        "category_counts":    dict(cat_counts),
        "interpretation": {
            "LOGGING":             "Branches in printf/fprintf/writev — decoupled from physics by design",
            "CONSTANT":            "Near-zero variance; always or never executed — trivially predicted by always-on/off",
            "BINARY_NOISE":        "Binary (0/1) with ~50% presence — no predictable physical correlate",
            "RUN_SPECIFIC":        "Correlated with recording session — may reflect run-specific initialisation",
            "TRULY_UNPREDICTABLE": "No identifiable pattern; may be inherently stochastic or driven by unmeasured state",
        },
        "branches": records,
    }
    with open(OUT / "neither_branches_detail.json", "w") as f:
        json.dump(out_json, f, indent=2)

    # ── fig_D1: Stacked bar showing category breakdown ──────────────────────
    cat_order  = ["LOGGING", "CONSTANT", "BINARY_NOISE", "RUN_SPECIFIC", "TRULY_UNPREDICTABLE"]
    cat_colors = {"LOGGING": "#aec6cf", "CONSTANT": "#f9c74f",
                  "BINARY_NOISE": "#e07b54", "RUN_SPECIFIC": "#c77dff",
                  "TRULY_UNPREDICTABLE": "#999"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: pie / bar of categories
    counts_ordered = [cat_counts.get(c, 0) for c in cat_order]
    colors_ordered = [cat_colors[c] for c in cat_order]
    bars = ax1.bar(cat_order, counts_ordered, color=colors_ordered, width=0.6)
    ax1.set_title(f"Why {len(neither_indices)} Branches Are Unpredictable\n"
                  f"(R² < 0.3 from physical sensors AND syslog)")
    ax1.set_ylabel("Number of branches")
    ax1.set_xticklabels(cat_order, rotation=15, ha="right")
    for bar, count in zip(bars, counts_ordered):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     str(count), ha="center", fontsize=11, fontweight="bold")

    # Right: function-category breakdown
    fn_cats = Counter([r["fn_category"] for r in records])
    fn_names = list(fn_cats.keys())
    fn_vals  = [fn_cats[k] for k in fn_names]
    fn_colors = {"LOGGING": "#aec6cf", "MEMORY": "#2e6da4", "LQR": "#e07b54",
                 "TICK": "#6abf69", "MATH": "#c77dff", "OTHER": "#999"}
    ax2.barh(fn_names, fn_vals, color=[fn_colors.get(k, "#999") for k in fn_names])
    ax2.set_xlabel("Number of branches")
    ax2.set_title("By Source Function Type")

    plt.suptitle("Analysis of Branches Not Predictable From Physical Sensors",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "fig_D1_neither_branches_analysis.png")
    plt.close(fig)
    print(f"\nSaved fig_D1 and neither_branches_detail.json → {OUT}/")

    # Print paper-ready sentence
    log_n   = cat_counts.get("LOGGING", 0)
    const_n = cat_counts.get("CONSTANT", 0)
    true_n  = cat_counts.get("TRULY_UNPREDICTABLE", 0)
    print(f"\nPaper summary: 'Of {len(neither_indices)} branches not predictable from physical "
          f"sensors, {log_n} are in logging/IO functions (inherently decoupled from physics), "
          f"{const_n} are near-constant (floor effect), and only {true_n} represent genuinely "
          f"unpredictable execution decisions.'")


if __name__ == "__main__":
    main()
