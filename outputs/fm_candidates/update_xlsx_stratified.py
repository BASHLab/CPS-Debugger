"""Add (or refresh) a 'Stratified 100K' sheet in cps_fm_candidates.xlsx with
the latest stratified-100K test metrics. Same row/column layout as the
'CPS FM to Trace Gen' (30K random) sheet so columns line up; only the data
sources change to results/<variant>_strat100k_fold_<f>/...json.

If a fold's stratified JSON is missing (still running), its values are left
empty for that row and "(running)" is appended to the cell. Variants with no
JSONs at all (BL-3b, MDLM during initial roll-out) get an "N/A — running"
placeholder.
"""
from pathlib import Path
import json
from typing import Optional

import numpy as np
import openpyxl
from openpyxl.styles import Font

XLSX = Path("/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates/results/cps_fm_candidates.xlsx")
RESULTS = Path("/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates/results")
SOURCE_SHEET = "CPS FM to Trace Gen"
TARGET_SHEET = "Stratified 100K"

ROW_MAP = {
    4: ("mdlm_strat100k_fold_{fold}/mdlm_eval_test.json",            None),
    5: ("ar_modern_strat100k_fold_{fold}/ar_modern_eval_test.json",  "timing_ar_modern_fold_0/ar_modern_eval_test.json"),
    6: ("bl3_strat100k_fold_{fold}/bl3_eval_test.json",              "timing_bl3_fold_0/bl3_eval_test.json"),
    # BL-3b: skipped (val 0.524 vs BL-3 0.075 = ~7× = severe overfit; standard
    # rank-16 LoRA on all-layers attention is too much capacity for ~2 train
    # sessions of correlated data — see paper for narration.)
    7: (None,                                                        None),
    8: ("bl4_strat100k_fold_{fold}/bl4_eval_test.json",              "timing_bl4_fold_0/bl4_eval_test.json"),
    9: ("bl6_strat100k_fold_{fold}/bl6_eval_test.json",              "timing_bl6_fold_0/bl6_eval_test.json"),
    # BL-7: sensor-only JEPA encoder pretrained from scratch (41 M params, ViT-Small,
    # conv frontend + axial-attention, K=4 dense tokens, mean-pool stopgap to AR-modern
    # decoder). No video. No timing JSON yet (would run via timing_bl7 in a separate pass).
    10: ("ar_modern_bl7_strat100k_fold_{fold}/ar_modern_bl7_eval_test.json", None),
    # 2x2 factorial Cell A: cross-attn decoder (Tier 1A.1) + v1 JEPA embeddings.
    11: ("ar_modern_bl7_cross_strat100k_fold_{fold}/ar_modern_bl7_cross_eval_test.json", None),
    # 2x2 factorial Cell B: prefix decoder + v2 embeddings (JEPA + class-balanced contrastive).
    12: ("ar_modern_bl7_v2_strat100k_fold_{fold}/ar_modern_bl7_v2_eval_test.json", None),
    # 2x2 factorial Cell C: cross-attn decoder + v2 embeddings (both Tier 1A.1 + Tier 2).
    13: ("ar_modern_bl7_v2_cross_strat100k_fold_{fold}/ar_modern_bl7_v2_cross_eval_test.json", None),
}
# MDLM timing JSON exists at timing_mdlm_fold_0/mdlm_eval_test.json
ROW_MAP[4] = (ROW_MAP[4][0], "timing_mdlm_fold_0/mdlm_eval_test.json")

COLUMN_SPEC = [
    ("F", ("ned_mean", "state", 0), "{:.4f}", "lower",   False),
    ("G", ("ned_mean", "state", 1), "{:.4f}", "lower",   False),
    ("H", ("ned_mean", "state", 2), "{:.4f}", "lower",   False),
    ("I", ("ned_mean", "all",   None), "{:.4f}", "lower",  False),
    ("J", ("exact_match", "state", 0), "{:.2f}%", "higher", True),
    ("K", ("exact_match", "state", 1), "{:.2f}%", "higher", True),
    ("L", ("exact_match", "state", 2), "{:.2f}%", "higher", True),
    ("M", ("exact_match", "all",   None), "{:.2f}%", "higher", True),
    ("N", ("vendi_ratio_gen_over_ref", "state", 0), "{:.3f}", "near_one", False),
    ("O", ("vendi_ratio_gen_over_ref", "state", 1), "{:.3f}", "near_one", False),
    ("P", ("vendi_ratio_gen_over_ref", "state", 2), "{:.3f}", "near_one", False),
    ("Q", ("vendi_ratio_gen_over_ref", "all",   None), "{:.3f}", "near_one", False),
    ("R", ("crystal_bleu",         "all", None), "{:.4f}",   "higher", False),
    ("S", ("self_bleu_generated",  "all", None), "{:.4f}",   "lower",  False),
    ("T", ("rouge_l_f1_mean",      "all", None), "{:.4f}",   "higher", False),
    ("U", ("token_acc_generation", "all", None), "{:.2f}%",  "higher", True),
    ("V", ("p99_ms_per_tick",      "timing", None), "{:.3f}", "lower", False),
]


def _scalar(j, key, mode_prefixes=("sampled_", "baseline_", "greedy_", "")):
    if j is None: return None
    for p in mode_prefixes:
        k = f"{p}{key}"
        if k in j and isinstance(j[k], (int, float)):
            return float(j[k])
    return None


def _per_state(j, key, state, mode_prefixes=("sampled_", "baseline_", "greedy_", "")):
    if j is None: return None
    for p in mode_prefixes:
        ps_key = f"{p}per_state"
        if ps_key in j and isinstance(j[ps_key], dict):
            sd = j[ps_key].get(str(state))
            if sd and key in sd and isinstance(sd[key], (int, float)):
                return float(sd[key])
    return None


def _try_load(path):
    if not path.exists(): return None
    try: return json.loads(path.read_text())
    except Exception: return None


def _agg_cell(values, fmt, percent_x100):
    arr = np.array([v for v in values if v is not None], dtype=float)
    if len(arr) == 0:
        return None
    if percent_x100: arr = arr * 100.0
    return f"{fmt.format(float(np.median(arr)))} [{fmt.format(float(arr.min()))}-{fmt.format(float(arr.max()))}]"


def _single_cell(value, fmt, percent_x100):
    if value is None: return None
    v = value * 100.0 if percent_x100 else value
    return fmt.format(v)


def _load_metric_for_row(pattern, timing_path, metric_spec):
    key, kind, state = metric_spec
    if kind == "timing":
        if timing_path is None: return []
        tj = _try_load(RESULTS / timing_path)
        v = _scalar(tj, key)
        return [v] if v is not None else []
    if pattern is None: return []
    fold_jsons = [_try_load(RESULTS / pattern.format(fold=f)) for f in range(5)]
    if kind == "all":
        return [_scalar(j, key) for j in fold_jsons]
    if kind == "state":
        return [_per_state(j, key, state) for j in fold_jsons]
    return []


def _parse_median(cell_text):
    if not cell_text or "N/A" in cell_text or "running" in cell_text:
        return None
    s = cell_text.replace("%", "").strip()
    if "[" in s: s = s.split("[", 1)[0].strip()
    try: return float(s)
    except ValueError: return None


def _best_index(values_text, direction):
    nums = [(_parse_median(t), i) for i, t in enumerate(values_text)]
    nums = [(n, i) for n, i in nums if n is not None]
    if not nums: return None
    if direction == "lower":   return min(nums)[1]
    if direction == "higher":  return max(nums, key=lambda t: t[0])[1]
    if direction == "near_one":return min(nums, key=lambda t: abs(t[0] - 1.0))[1]
    return None


def _count_folds_present(pattern):
    """How many of the 5 fold JSONs exist."""
    if pattern is None: return 0
    return sum(1 for f in range(5) if (RESULTS / pattern.format(fold=f)).exists())


def main():
    wb = openpyxl.load_workbook(XLSX)
    # In-place update: if the stratified sheet exists, refresh it directly.
    # If it doesn't, bootstrap from the legacy 30K sheet (if still present).
    if TARGET_SHEET in wb.sheetnames:
        ws = wb[TARGET_SHEET]
    else:
        if SOURCE_SHEET not in wb.sheetnames:
            raise SystemExit(
                f"Neither '{TARGET_SHEET}' nor '{SOURCE_SHEET}' is present. "
                "Cannot bootstrap — provide a workbook with one of them."
            )
        src = wb[SOURCE_SHEET]
        ws = wb.copy_worksheet(src)
        ws.title = TARGET_SHEET

    # Update top-line note (row 1) to reflect stratified
    note = ("Stratified 100K test eval — ALL state-0 + ALL state-2 + ~80K state-1 "
            "(deterministic via EVAL_SUBSET_SEED). state-0 ≈ 17,581; state-1 ≈ 80,000; "
            "state-2 ≈ 3,007; total ≈ 100,588 per fold. Timing pass uses n=1000 on "
            "H100, single fold. All cells: median [min-max] across the 5 CV folds.")
    ws.cell(row=1, column=1).value = note

    # Build cells
    # Variants we deliberately skipped (vs still-running)
    SKIPPED = {
        7: "N/A — overfit (val 0.524 vs BL-3 0.075; LoRA rank-16/all-layers too much capacity)",
    }

    rows_data = {}
    fold_status = {}
    for excel_row, (pattern, timing_path) in ROW_MAP.items():
        n_folds = _count_folds_present(pattern)
        fold_status[excel_row] = n_folds
        cells = {}
        if n_folds == 0:
            label = SKIPPED.get(excel_row, "N/A — running")
            for col_letter, _, _, _, _ in COLUMN_SPEC:
                cells[col_letter] = label
            # Timing column might still load even without metric folds
            for col_letter, mspec, fmt, _, percent_x100 in COLUMN_SPEC:
                key, kind, _state = mspec
                if kind == "timing":
                    vals = _load_metric_for_row(pattern, timing_path, mspec)
                    if vals:
                        cells[col_letter] = _single_cell(vals[0], fmt, percent_x100) or label
        else:
            for col_letter, mspec, fmt, _, percent_x100 in COLUMN_SPEC:
                key, kind, _state = mspec
                vals = _load_metric_for_row(pattern, timing_path, mspec)
                if kind == "timing":
                    text = _single_cell(vals[0] if vals else None, fmt, percent_x100)
                else:
                    text = _agg_cell(vals, fmt, percent_x100)
                if text is None:
                    text = ""
                if 0 < n_folds < 5 and kind != "timing":
                    text = f"{text} ({n_folds}/5)"
                cells[col_letter] = text
        rows_data[excel_row] = cells

    # Write cells (clear bold first)
    for excel_row, cells in rows_data.items():
        for col_letter, text in cells.items():
            cell = ws[f"{col_letter}{excel_row}"]
            cell.value = text
            f = cell.font
            cell.font = Font(name=f.name, size=f.size, bold=False, italic=f.italic,
                             color=f.color, vertAlign=f.vertAlign, underline=f.underline,
                             strike=f.strike)

    # Bold best per column (skips N/A and partial rows)
    for col_letter, _, _, direction, _ in COLUMN_SPEC:
        col_values = [rows_data[r][col_letter] for r in sorted(rows_data)]
        best_local = _best_index(col_values, direction)
        if best_local is None: continue
        excel_row = sorted(rows_data)[best_local]
        cell = ws[f"{col_letter}{excel_row}"]
        f = cell.font
        cell.font = Font(name=f.name, size=f.size, bold=True, italic=f.italic,
                         color=f.color, vertAlign=f.vertAlign, underline=f.underline,
                         strike=f.strike)

    # Bold headers (rows 1-3) — preserved by copy_worksheet style
    for r in (1, 2, 3):
        for c in range(1, ws.max_column + 1):
            cell = ws.cell(row=r, column=c)
            if cell.value is None: continue
            f = cell.font
            cell.font = Font(name=f.name, size=f.size, bold=True, italic=f.italic,
                             color=f.color, vertAlign=f.vertAlign, underline=f.underline,
                             strike=f.strike)

    wb.save(XLSX)
    print(f"Saved {XLSX}")
    print(f"\nFolds-present per variant (row, ID, n_folds):")
    for r in sorted(rows_data):
        ident = ws.cell(row=r, column=1).value
        sample = ws.cell(row=r, column=ord("I") - ord("A") + 1).value
        print(f"  row {r} ({ident}): {fold_status[r]}/5 folds; NED-all = {sample!r}")


if __name__ == "__main__":
    main()
