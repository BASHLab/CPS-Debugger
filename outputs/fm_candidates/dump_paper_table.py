"""Dump 5-fold cross-variant test results to a paper-ready Markdown file.

For each (variant, metric) cell we report median + [min, max] over the 5 folds.
Includes both an "all" (overall) column and a "worst-state" column for the
three FSM states (worst defined per metric: max NED / min exact_match /
max |vendi_ratio - 1|).

Also writes a JSON sidecar with the same data for downstream re-styling.
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


VARIANTS = [
    ("MDLM",                  "mdlm_fold_{fold}/mdlm_eval_{split}.json"),
    ("AR-modern",             "ar_modern_fold_{fold}/ar_modern_eval_{split}.json"),
    ("BL-3 (Chronos+VJEPA)",  "bl3_fold_{fold}/bl3_eval_{split}.json"),
    ("BL-4 (Chronos+CoTrk)",  "bl4_fold_{fold}/bl4_eval_{split}.json"),
    ("BL-6 (MOMENT)",         "bl6_fold_{fold}/bl6_eval_{split}.json"),
]
FLOORS = [
    ("unigram",       "baselines_fold_{fold}/unigram_eval_{split}.json"),
    ("copy_modal",    "baselines_fold_{fold}/copy_modal_eval_{split}.json"),
    ("copy_previous", "baselines_fold_{fold}/copy_previous_eval_{split}.json"),
]
METRICS = [
    # (key, label, lower_is_better, fmt)
    ("ned_mean",                 "NED",        True,  "{:.4f}"),
    ("exact_match",              "exact (%)",  False, "{:.2%}"),
    ("vendi_ratio_gen_over_ref", "Vendi-ratio", None, "{:.3f}"),  # closer-to-1 is better
    ("crystal_bleu",             "CrystalBLEU", False, "{:.4f}"),
    ("self_bleu_generated",      "Self-BLEU_gen", True, "{:.4f}"),
]
TIMING = ("median_ms_per_tick", "ms/tick", True, "{:.2f}")


def _try_load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _scalar(j, key, mode_prefixes=("sampled_", "baseline_", "greedy_", "")):
    """Resolve a metric key by trying mode-prefixed variants in order.

    Includes "baseline_" so trivial-floor JSONs (which use baseline_* keys
    instead of sampled_* / greedy_*) resolve correctly.
    """
    if j is None:
        return None
    for p in mode_prefixes:
        k = f"{p}{key}"
        if k in j and isinstance(j[k], (int, float)):
            return float(j[k])
    return None


def _per_state(j, key, state, mode_prefixes=("sampled_", "baseline_", "greedy_", "")):
    """Resolve a per-state metric key for one FSM state."""
    if j is None:
        return None
    for p in mode_prefixes:
        ps_key = f"{p}per_state"
        if ps_key in j and isinstance(j[ps_key], dict):
            sd = j[ps_key].get(str(state))
            if sd and key in sd and isinstance(sd[key], (int, float)):
                return float(sd[key])
    return None


def _agg(values: list[float]) -> dict[str, Optional[float]]:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if len(arr) == 0:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": float(np.median(arr)), "min": float(arr.min()),
            "max": float(arr.max()), "n": int(len(arr))}


def _fmt_cell(stats: dict, fmt: str) -> str:
    if stats["median"] is None:
        return "—"
    return f"{fmt.format(stats['median'])} [{fmt.format(stats['min'])}, {fmt.format(stats['max'])}]"


def _worst_state_value(j, key: str, lower_is_better, mode_prefixes=("sampled_", "greedy_", "")):
    """Return the worst per-state value for this metric in one fold's JSON.

    For NED / Self-BLEU_gen (lower is better): worst = max.
    For exact / CrystalBLEU (higher is better): worst = min.
    For Vendi-ratio (closer-to-1 is better): worst = arg max |v - 1|.
    """
    vals = []
    for state in (0, 1, 2):
        v = _per_state(j, key, state, mode_prefixes)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    if lower_is_better is True:
        return max(vals)
    if lower_is_better is False:
        return min(vals)
    # closer-to-1 metric (Vendi-ratio)
    return vals[int(np.argmax([abs(v - 1.0) for v in vals]))]


def _gather_variant(variant_label: str, pat: str, result_dir: Path,
                    split: str) -> dict:
    """Load all 5 fold JSONs and compute median+[min,max] across folds for
    each metric / per-state / worst-state combination."""
    fold_jsons = []
    for fold in range(5):
        p = result_dir / pat.format(fold=fold, split=split)
        fold_jsons.append(_try_load(p))

    out = {"variant": variant_label, "fold_paths": [str(result_dir / pat.format(fold=fold, split=split)) for fold in range(5)]}
    out["fold_jsons_present"] = [j is not None for j in fold_jsons]

    for key, _, lower, _ in METRICS:
        # Overall
        overall = _agg([_scalar(j, key) for j in fold_jsons])
        out[f"{key}__all"] = overall
        # Per-state
        for s in (0, 1, 2):
            ps = _agg([_per_state(j, key, s) for j in fold_jsons])
            out[f"{key}__state{s}"] = ps
        # Worst state across folds
        ws = _agg([_worst_state_value(j, key, lower) for j in fold_jsons])
        out[f"{key}__worst_state"] = ws

    # Timing — overall only (per-state timing isn't well-defined per the pipeline)
    tk, _, _, _ = TIMING
    out["timing__all"] = _agg([_scalar(j, tk) for j in fold_jsons])
    return out


def _md_table_rows(rows, columns):
    """columns: list of (header, key) pairs. rows: list of dicts."""
    header = "| Variant | " + " | ".join(h for h, _ in columns) + " |"
    sep    = "|" + "|".join(["---"] * (len(columns) + 1)) + "|"
    body   = []
    for r in rows:
        cells = [r["variant"]]
        for _, k in columns:
            v = r.get(k)
            if v is None:
                cells.append("—")
            else:
                cells.append(v)
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates/results")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-md", default="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates/results/paper_table_test.md")
    ap.add_argument("--out-json", default="/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates/results/paper_table_test.json")
    ap.add_argument("--header", default="5-fold test results (30K random subset per fold)")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)

    raw_models = [_gather_variant(label, pat, result_dir, args.split)
                  for label, pat in VARIANTS]
    raw_floors = [_gather_variant(label, pat, result_dir, args.split)
                  for label, pat in FLOORS]

    # Build display rows for the Markdown table
    def fmt_row(r):
        out = {"variant": r["variant"]}
        for key, _, _, fmt in METRICS:
            out[f"{key}__all"] = _fmt_cell(r[f"{key}__all"], fmt)
            out[f"{key}__worst"] = _fmt_cell(r[f"{key}__worst_state"], fmt)
        tk, _, _, tfmt = TIMING
        out["timing__all"] = _fmt_cell(r["timing__all"], tfmt)
        return out

    rows_models = [fmt_row(r) for r in raw_models]
    rows_floors = [fmt_row(r) for r in raw_floors]

    # Headline (overall, all-states) table — no worst-state column per user request
    headline_cols = [(label, f"{key}__all") for key, label, _, _ in METRICS]
    headline_cols.append((f"{TIMING[1]}", "timing__all"))

    # Per-state table — by state, NED + exact + Vendi-ratio only
    def per_state_rows(raw_rows, state):
        rs = []
        for r in raw_rows:
            row = {"variant": r["variant"]}
            for key, label, _, fmt in METRICS:
                if key not in ("ned_mean", "exact_match", "vendi_ratio_gen_over_ref"):
                    continue
                row[f"{key}__state"] = _fmt_cell(r[f"{key}__state{state}"], fmt)
            rs.append(row)
        return rs

    per_state_cols = [(label, f"{k}__state") for k, label, _, _ in METRICS
                      if k in ("ned_mean", "exact_match", "vendi_ratio_gen_over_ref")]

    md = []
    md.append(f"# {args.header}\n")
    md.append("Format per cell: **median [min, max]** across the 5 folds.\n")

    md.append("\n## Headline (across-states aggregate)\n")
    md.append("Models:\n")
    md.append(_md_table_rows(rows_models, headline_cols))
    md.append("\n\nFloor (trivial baselines):\n")
    md.append(_md_table_rows(rows_floors, headline_cols))

    for state, label in [(0, "FSM state 0 (start-up phase, ~1.7% of corpus)"),
                          (1, "FSM state 1 (steady-state, ~97% of corpus)"),
                          (2, "FSM state 2 (transient, ~0.6% of corpus)")]:
        md.append(f"\n## Per-state — {label}\n")
        md.append("Models:\n")
        md.append(_md_table_rows(per_state_rows(raw_models, state), per_state_cols))
        md.append("\n\nFloor:\n")
        md.append(_md_table_rows(per_state_rows(raw_floors, state), per_state_cols))

    md_text = "\n".join(md) + "\n"
    Path(args.out_md).write_text(md_text)
    print(f"Wrote {args.out_md}")

    # JSON sidecar (raw stats — easier downstream)
    Path(args.out_json).write_text(json.dumps(
        {"split": args.split, "models": raw_models, "floors": raw_floors},
        indent=2, default=lambda o: o if isinstance(o, (int, float, str, list, dict, type(None))) else str(o),
    ))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
