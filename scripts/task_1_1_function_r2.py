"""
task_1_1_function_r2.py — Per-function R² within the BALANCE state.

Uses stratified_r2.csv (already computed within-BALANCE R² per branch) and
groups branches by parent function to answer: is the fine-grained trace-recovery
claim driven by control logic (pou_lqr_sim/tick), data-movement (memcpy), or
logging infrastructure?

Output:
  outputs/phase3/function_r2_summary.json
  outputs/phase3/fig_function_r2_groups.png
  outputs/phase3/fig_function_r2_branches.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT   = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
P2_OUT = ROOT / "outputs/phase2"
P3_OUT = ROOT / "outputs/phase3"
P3_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.dpi": 300})

# ── function grouping ─────────────────────────────────────────────────────────
# Maps function name keywords → canonical group label
FUNC_GROUPS = {
    "tick":              "tick",
    "execute":           "tick",
    "stays_balanced":    "tick",
    "pou_main":          "tick",
    "pou_general":       "tick",
    "turn_on":           "tick",
    "turn_off":          "tick",
    "pou_lqr":           "pou_lqr_sim",
    "lqr":               "pou_lqr_sim",
    "reference_pt":      "pou_lqr_sim",
    "compute_force":     "pou_lqr_sim",
    "compute_velocity":  "pou_lqr_sim",
    "standup":           "pou_standup_rel",
    "pou_standu":        "pou_standup_rel",
    "memcpy":            "memcpy",
    "memset":            "memset",
    "sin":               "math_lib",
    "pio2":              "math_lib",
    "fmin":              "math_lib",
    "fmax":              "math_lib",
    "log_data":          "logging",
    "printf":            "logging",
    "vfprintf":          "logging",
    "fprintf":           "logging",
    "write":             "logging",
    "fwrite":            "logging",
    "puts":              "logging",
    "writev":            "logging",
}

GROUP_COLORS = {
    "tick":              "#2196F3",
    "pou_lqr_sim":       "#4CAF50",
    "pou_standup_rel":   "#FF9800",
    "memcpy":            "#9C27B0",
    "memset":            "#00BCD4",
    "math_lib":          "#F44336",
    "logging":           "#9E9E9E",
    "other":             "#FFEB3B",
}

GROUP_ORDER = ["pou_lqr_sim", "tick", "memcpy", "memset", "math_lib",
               "pou_standup_rel", "logging", "other"]


def assign_group(fn_name: str) -> str:
    fn_lower = fn_name.lower()
    for kw, grp in FUNC_GROUPS.items():
        if kw in fn_lower:
            return grp
    return "other"


def main():
    # ── load stratified_r2.csv ────────────────────────────────────────────────
    csv_path = P2_OUT / "stratified_r2.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run phase2_A_stratified_analysis.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} branches from stratified_r2.csv")

    # Load fn_names from vocab_info for accurate function name resolution
    vocab_path = ROOT / "outputs/experiments/vocab_info.json"
    fn_names = {}
    if vocab_path.exists():
        vinfo = json.loads(vocab_path.read_text())
        fn_names = vinfo.get("fn_names", {})
        # fn_names maps func_id (str) → function name
        labels  = vinfo.get("labels", {})  # branch_id → label string "fname:src→dst"

    # Resolve function name per branch from branch_id "func_id:src_pc:dst_pc"
    def branch_fn_name(branch_id):
        if branch_id in labels:
            return labels[branch_id].split(":")[0]
        func_id = branch_id.split(":")[0]
        return fn_names.get(func_id, f"func_{func_id}")

    df["fn_name"] = df["branch_id"].apply(branch_fn_name)
    df["group"]   = df["fn_name"].apply(assign_group)

    # ── per-group stats ───────────────────────────────────────────────────────
    group_stats = (
        df.groupby("group")["r2_within_balance"]
        .agg(["mean", "std", "count",
              lambda x: (x > 0.2).sum(),
              lambda x: (x > 0.3).sum()])
        .rename(columns={
            "mean": "mean_r2", "std": "std_r2", "count": "n_branches",
            "<lambda_0>": "n_r2_gt_02", "<lambda_1>": "n_r2_gt_03"
        })
        .reset_index()
    )
    group_stats["frac_r2_gt_02"] = group_stats["n_r2_gt_02"] / group_stats["n_branches"]
    group_stats["frac_r2_gt_03"] = group_stats["n_r2_gt_03"] / group_stats["n_branches"]

    # Reorder by GROUP_ORDER
    group_stats["_order"] = group_stats["group"].map(
        {g: i for i, g in enumerate(GROUP_ORDER)}
    ).fillna(99)
    group_stats = group_stats.sort_values("_order").drop(columns="_order")

    print("\n── Per-Function Group Within-BALANCE R² ──")
    print(group_stats[["group", "n_branches", "mean_r2", "std_r2",
                        "frac_r2_gt_03"]].to_string(index=False, float_format="%.3f"))

    # ── save summary JSON ─────────────────────────────────────────────────────
    out = {
        "n_branches_total": int(len(df)),
        "overall_mean_within_balance_r2": float(df["r2_within_balance"].mean()),
        "groups": [],
    }
    for _, row in group_stats.iterrows():
        grp_branches = df[df["group"] == row["group"]]
        top5 = grp_branches.nlargest(5, "r2_within_balance")[
            ["branch_id", "fn_name", "r2_within_balance"]
        ].to_dict("records")
        out["groups"].append({
            "group":          row["group"],
            "n_branches":     int(row["n_branches"]),
            "mean_r2":        float(row["mean_r2"]),
            "std_r2":         float(row["std_r2"]),
            "frac_r2_gt_02":  float(row["frac_r2_gt_02"]),
            "frac_r2_gt_03":  float(row["frac_r2_gt_03"]),
            "top5_branches":  [
                {k: (float(v) if isinstance(v, float) else v) for k, v in b.items()}
                for b in top5
            ],
        })
    (P3_OUT / "function_r2_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved function_r2_summary.json")

    # ── KEY DECISION OUTPUT ───────────────────────────────────────────────────
    lqr_r2  = group_stats.loc[group_stats["group"] == "pou_lqr_sim", "mean_r2"].values
    tick_r2 = group_stats.loc[group_stats["group"] == "tick",         "mean_r2"].values
    mc_r2   = group_stats.loc[group_stats["group"] == "memcpy",       "mean_r2"].values
    print("\n── DECISION GATE ──")
    if len(lqr_r2):
        r = lqr_r2[0]
        print(f"pou_lqr_sim mean within-BALANCE R² = {r:.3f}", end="  ")
        if r > 0.2:
            print("→ PROCEED with full plan (fine-grained claim holds)")
        else:
            print("→ REFRAME: lqr branches not physically predictable within BALANCE")
    if len(mc_r2):
        print(f"memcpy      mean within-BALANCE R² = {mc_r2[0]:.3f}")
    if len(tick_r2):
        print(f"tick        mean within-BALANCE R² = {tick_r2[0]:.3f}")

    # ── Figure 1: Grouped bar chart ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(group_stats))
    colors = [GROUP_COLORS.get(g, "#888") for g in group_stats["group"]]
    bars = ax.bar(xs, group_stats["mean_r2"], color=colors,
                  yerr=group_stats["std_r2"] / np.sqrt(group_stats["n_branches"].clip(1)),
                  capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g}\n(n={n})" for g, n in
                        zip(group_stats["group"], group_stats["n_branches"])],
                       rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Mean Within-BALANCE R²")
    ax.set_title("Branch Predictability by Function Group (within BALANCE state)")
    ax.axhline(0.2, color="red", linestyle="--", linewidth=1, label="R²=0.2 threshold")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(P3_OUT / "fig_function_r2_groups.png")
    plt.close(fig)
    print("Saved fig_function_r2_groups.png")

    # ── Figure 2: Per-branch bar chart sorted by R², coloured by group ────────
    df_sorted = df.sort_values("r2_within_balance", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(16, 5))
    bar_colors = [GROUP_COLORS.get(g, "#888") for g in df_sorted["group"]]
    ax.bar(range(len(df_sorted)), df_sorted["r2_within_balance"],
           color=bar_colors, edgecolor="none")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axhline(0.3, color="red", linestyle="--", linewidth=1, label="R²=0.3")
    ax.set_xlabel("Branch rank (sorted by within-BALANCE R²)")
    ax.set_ylabel("Within-BALANCE R²")
    ax.set_title("Per-Branch Within-BALANCE R², colored by function group")
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()
                       if g in df_sorted["group"].values]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(P3_OUT / "fig_function_r2_branches.png")
    plt.close(fig)
    print("Saved fig_function_r2_branches.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
