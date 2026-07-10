"""Dump 5-fold cross-variant test results to a paper-ready CSV.

Format mirrors the user's `cps_fm_candidates` template:
  cols 1-5: ID, Sensor Encoder, Video Encoder, Fusion, Trace Generator
  cols 6+:  per-metric × {state 0, state 1, state 2, all}
  cells:    median [min-max] across the 5 folds (timing is single-fold)

Two header rows before the data:
  row 1: metric group names (each spans 4 columns; timing is single)
  row 2: per-column subhead (state and approx n=... in parentheses)
  row 3: standard variant-identity column names

H100-pinned timing comes from the dedicated timing-only pass
(`results/timing_<variant>_fold_0/<variant>_eval_test.json`).
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np


# ── Variant catalogue (matches the user's CSV template) ────────────────────
# (id, sensor_encoder, video_encoder, fusion, trace_generator, json_pattern,
#  timing_json_path)
VARIANTS = [
    ("BL1", "Chronos-2", "None", "Concat", "MDLM",
     "mdlm_fold_{fold}/mdlm_eval_{split}.json",
     "timing_mdlm_fold_0/mdlm_eval_test.json"),
    ("BL2", "Chronos-2", "None", "Concat", "AR-modern",
     "ar_modern_fold_{fold}/ar_modern_eval_{split}.json",
     "timing_ar_modern_fold_0/ar_modern_eval_test.json"),
    ("BL3", "Chronos-2", "V-JEPA 2.1 (frozen)", "Concat", "AR-modern",
     "bl3_fold_{fold}/bl3_eval_{split}.json",
     "timing_bl3_fold_0/bl3_eval_test.json"),
    ("BL3b", "Chronos-2", "V-JEPA 2.1 (LoRA, fused-qkv)", "Concat", "AR-modern",
     None,    # eval skipped — val 0.524 vs BL3 0.075 = overfit
     None),
    ("BL4", "Chronos-2", "CoTracker3 (online, full frame)", "Concat", "AR-modern",
     "bl4_fold_{fold}/bl4_eval_{split}.json",
     "timing_bl4_fold_0/bl4_eval_test.json"),
    ("BL5", "Chronos-2", "V-JEPA 2.1 + CoTracker3", "Concat", "AR-modern",
     None,    # skipped per decision rule (no video variant beat AR-modern)
     None),
    ("BL6", "MOMENT-1", "None", "Concat", "AR-modern",
     "bl6_fold_{fold}/bl6_eval_{split}.json",
     "timing_bl6_fold_0/bl6_eval_test.json"),
]

# Trivial-floor baselines (separate section below)
FLOORS = [
    ("Floor-unigram",       "—", "—", "—", "trivial unigram histogram",
     "baselines_fold_{fold}/unigram_eval_{split}.json", None),
    ("Floor-copy_modal",    "—", "—", "—", "always argmax(token freq)",
     "baselines_fold_{fold}/copy_modal_eval_{split}.json", None),
    ("Floor-copy_previous", "—", "—", "—", "x[t] = x[t-1]",
     "baselines_fold_{fold}/copy_previous_eval_{split}.json", None),
]

# Metrics that ARE computed per-FSM-state (compute_per_state_metrics in metrics.py).
# These get 4 sub-columns: state 0 / state 1 / state 2 / all.
METRIC_PERSTATE = [
    ("ned_mean",                 "NED (median [min-max])",                  "{:.4f}"),
    ("exact_match",              "Exact Match % (median [min-max])",        "{:.2%}"),
    ("vendi_ratio_gen_over_ref", "Vendi-ratio (ideal=1, median [min-max])", "{:.3f}"),
]

# Metrics computed only on the full sample (no per-state breakout) + timing.
# Each gets a single all-set column.
METRIC_SINGLE = [
    ("crystal_bleu",         "CrystalBLEU (median [min-max], all)",         "{:.4f}"),
    ("self_bleu_generated",  "Self-BLEU_gen (median [min-max], all)",       "{:.4f}"),
    ("rouge_l_f1_mean",      "ROUGE-L F1 (median [min-max], all)",          "{:.4f}"),
    ("token_acc_generation", "Token-acc gen (median [min-max], all)",       "{:.4f}"),
    ("p99_ms_per_tick",      "Inference Time (99th pct ms/tick, H100, n=1000)", "{:.3f}"),
]


def _try_load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _scalar(j, key, mode_prefixes=("sampled_", "baseline_", "greedy_", "")) -> Optional[float]:
    if j is None:
        return None
    for p in mode_prefixes:
        k = f"{p}{key}"
        if k in j and isinstance(j[k], (int, float)):
            return float(j[k])
    return None


def _per_state(j, key, state, mode_prefixes=("sampled_", "baseline_", "greedy_", "")) -> Optional[float]:
    if j is None:
        return None
    for p in mode_prefixes:
        ps_key = f"{p}per_state"
        if ps_key in j and isinstance(j[ps_key], dict):
            sd = j[ps_key].get(str(state))
            if sd and key in sd and isinstance(sd[key], (int, float)):
                return float(sd[key])
    return None


def _per_state_n(j, state, mode_prefixes=("sampled_", "baseline_", "greedy_", "")) -> Optional[int]:
    if j is None:
        return None
    for p in mode_prefixes:
        ps_key = f"{p}per_state"
        if ps_key in j and isinstance(j[ps_key], dict):
            sd = j[ps_key].get(str(state))
            if sd and "n" in sd:
                return int(sd["n"])
    return None


def _agg_cell(values, fmt) -> str:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if len(arr) == 0:
        return ""
    med = float(np.median(arr))
    lo  = float(arr.min())
    hi  = float(arr.max())
    return f"{fmt.format(med)} [{fmt.format(lo)}-{fmt.format(hi)}]"


# ── "Best in column" comparison (per metric direction) ────────────────────
# direction: "lower" / "higher" / "near_one"
METRIC_DIRECTION = {
    "ned_mean":                 "lower",
    "exact_match":              "higher",
    "vendi_ratio_gen_over_ref": "near_one",
    "crystal_bleu":             "higher",
    "self_bleu_generated":      "lower",
    "rouge_l_f1_mean":          "higher",
    "token_acc_generation":     "higher",
    "p99_ms_per_tick":          "lower",
}


def _parse_median(cell: str) -> Optional[float]:
    """Pull the median (the leading number) out of a 'X [Y-Z]' cell."""
    if not cell or cell == "N/A":
        return None
    cell = cell.strip()
    # Strip optional bold markers used by other code paths (defensive)
    cell = cell.replace("**", "")
    # Take everything before " [" or before " " before "[", or whole if no bracket.
    if "[" in cell:
        cell = cell.split("[", 1)[0]
    cell = cell.strip().rstrip("%").strip()
    try:
        return float(cell)
    except ValueError:
        return None


def _best_index(values, direction):
    nums = [(_parse_median(v), i) for i, v in enumerate(values)]
    nums = [(n, i) for n, i in nums if n is not None]
    if not nums:
        return None
    if direction == "lower":
        return min(nums)[1]
    if direction == "higher":
        return max(nums, key=lambda t: t[0])[1]
    if direction == "near_one":
        return min(nums, key=lambda t: abs(t[0] - 1.0))[1]
    return None


def _row_for_variant(vid, sensor_enc, video_enc, fusion, gen, pattern, timing_path,
                     result_dir: Path, split: str):
    """Build one CSV data row for a variant (or N/A row if pattern is None)."""
    # If no JSON pattern, this variant was skipped — emit a row of "N/A".
    if pattern is None:
        empty_metric_cells = ["N/A"] * (len(METRIC_PERSTATE) * 4 + len(METRIC_SINGLE))
        return [vid, sensor_enc, video_enc, fusion, gen] + empty_metric_cells, None

    fold_jsons = [_try_load(result_dir / pattern.format(fold=f, split=split))
                  for f in range(5)]

    cells = []
    n_per_state = {0: [], 1: [], 2: []}

    for key, _, fmt in METRIC_PERSTATE:
        for state in (0, 1, 2):
            vals = [_per_state(j, key, state) for j in fold_jsons]
            cells.append(_agg_cell(vals, fmt))
            ns = [_per_state_n(j, state) for j in fold_jsons]
            for n in ns:
                if n is not None:
                    n_per_state[state].append(n)
        cells.append(_agg_cell([_scalar(j, key) for j in fold_jsons], fmt))

    # Single-column metrics: most come from the per-fold metric JSONs (median+[min-max]
    # across folds); the timing column uniquely comes from the H100 timing-only JSON.
    timing_j = _try_load(result_dir / timing_path) if timing_path else None
    for key, _, fmt in METRIC_SINGLE:
        if key == "p99_ms_per_tick":
            v = _scalar(timing_j, key)
            cells.append(fmt.format(v) if v is not None else "")
        else:
            vals = [_scalar(j, key) for j in fold_jsons]
            cells.append(_agg_cell(vals, fmt))

    return [vid, sensor_enc, video_enc, fusion, gen] + cells, n_per_state


def _avg_n(n_per_state_lists):
    """Average n per state across all variants for the n-count header row."""
    out = {}
    for s in (0, 1, 2):
        vals = []
        for d in n_per_state_lists:
            if d is not None:
                vals.extend(d.get(s, []))
        out[s] = int(round(float(np.mean(vals)))) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates/results")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-csv", default="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates/results/paper_table_test.csv")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)

    rows_models = []
    n_lists_models = []
    for tup in VARIANTS:
        row, n = _row_for_variant(*tup, result_dir=result_dir, split=args.split)
        rows_models.append(row); n_lists_models.append(n)

    avg_n = _avg_n(n_lists_models)
    n_total = sum(avg_n[s] for s in (0, 1, 2)) if all(avg_n.values()) else None

    # ── Bold best-in-column for each metric column over the model rows ──
    # Compute column-by-column. Identity cols (0..4) and N/A cells are skipped.
    n_id_cols = 5
    metric_keys_in_order = []
    for k, _, _ in METRIC_PERSTATE:
        metric_keys_in_order.extend([(k, "state0"), (k, "state1"), (k, "state2"), (k, "all")])
    for k, _, _ in METRIC_SINGLE:
        metric_keys_in_order.append((k, "all"))

    for col_offset, (key, _) in enumerate(metric_keys_in_order):
        col_idx = n_id_cols + col_offset
        col_values = [r[col_idx] for r in rows_models]
        best = _best_index(col_values, METRIC_DIRECTION.get(key, "higher"))
        if best is not None:
            rows_models[best][col_idx] = f"**{rows_models[best][col_idx]}**"

    # ── Header rows (all bolded with **) ──
    id_cols = ["**ID**", "**Sensor Encoder**", "**Video Encoder**", "**Fusion**", "**Trace Generator**"]

    # Row A: metric group names (each spans 4 cols for per-state, 1 for single)
    rowA = [""] * n_id_cols
    for _, label, _ in METRIC_PERSTATE:
        rowA.extend([f"**{label}**", "", "", ""])
    for _, label, _ in METRIC_SINGLE:
        rowA.append(f"**{label}**")

    # Row B: identity headers + state shorthand under per-state metrics
    rowB = id_cols + sum(
        [["**state 0**", "**state 1**", "**state 2**", "**all**"] for _ in METRIC_PERSTATE],
        []
    ) + [""] * len(METRIC_SINGLE)

    # Counts row at the very top (single, not repeated above each column)
    counts_msg = (f"Sample sizes per fold (median across 5 folds): "
                  f"state 0 ≈ {avg_n[0]}, state 1 ≈ {avg_n[1]}, state 2 ≈ {avg_n[2]}; "
                  f"all = {n_total}; timing pass uses n = 1000 on H100, single fold.")

    out_path = Path(args.out_csv)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([counts_msg])
        w.writerow([])     # blank separator row
        w.writerow(rowA)
        w.writerow(rowB)
        for r in rows_models:
            w.writerow(r)

    print(f"Wrote {out_path}")
    print(f"\n  state-0 avg n = {avg_n[0]}, state-1 avg n = {avg_n[1]}, "
          f"state-2 avg n = {avg_n[2]}, total ≈ {n_total}")
    print("  Bold (**...**) marks: header rows, identity column names, "
          "and the best variant in each metric column.")


if __name__ == "__main__":
    main()
