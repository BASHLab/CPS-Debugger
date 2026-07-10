"""Aggregate 5-fold eval JSONs into a single comparison table.

Layout on disk (one JSON per (variant, fold, split)):
  results/ar_fold_{i}/ar_eval_{split}.json
  results/ar_modern_fold_{i}/ar_modern_eval_{split}.json
  results/mdlm_fold_{i}/mdlm_eval_{split}.json
  results/baselines_fold_{i}/{unigram,copy_modal,copy_previous}_eval_{split}.json

For each variant we emit per-metric:
  median, min, max, raw five values (one per fold).

5-fold with mean±std is under-powered — we explicitly avoid that aggregation
and report median + [min, max] + raw values per the plan.

A "Floor" band lists trivial baselines. Any model metric whose median is
within 5% of the best floor median is flagged — that metric can't
discriminate on this corpus.

Conditional n-gram JS is bucketed by FSM state (pendulum_state: 0, 1, 2) and
reported both as a mean/worst summary per variant and as a per-state table.
State 1 dominates (~97% of ticks); states 0 and 2 are rare transients where
worst-case is the interesting signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results"


# Metrics to surface in the comparison table, in priority order.
# Some are only present with a mode prefix (e.g. "greedy_ned_mean" for AR);
# we auto-resolve by trying prefixed variants when the bare key is absent.
HEADLINE_METRICS = [
    "ned_mean",
    "exact_match",
    "vendi_ratio_gen_over_ref",
]

DIAGNOSTIC_METRICS = [
    "ned_median",
    "rouge_l_f1_mean",
    "token_acc_generation",
    "crystal_bleu",
    "bleu",
    "mauve",
    "distinct_1",
    "distinct_2",
    "self_bleu_generated",
    "vendi_generated",
    "vendi_reference",
    "token_kl_divergence",
]

TIMING_METRICS = [
    "median_ms_per_tick",
    "p90_ms_per_tick",
    "p99_ms_per_tick",
]


def _resolve(obj: Dict, key: str) -> Optional[float]:
    """Find `key` in a result JSON, trying bare then mode-prefixed variants.

    Models write metrics under mode prefixes (greedy_*, sampled_*); baselines
    under baseline_*. We prefer sampled_/baseline_ over greedy_ when both exist
    (for model-vs-baseline fairness: baselines aren't "greedy" or "sampled").
    """
    for probe in (key, f"sampled_{key}", f"baseline_{key}", f"greedy_{key}"):
        if probe in obj:
            return obj[probe]
    return None


def _load_variant(variant: str, file_pattern: str, result_dir: Path,
                  split: str) -> List[Dict]:
    """Load per-fold JSONs for a given variant; silently skip missing folds."""
    out = []
    for fold_idx in range(5):
        path = result_dir / file_pattern.format(fold=fold_idx, split=split)
        if path.exists():
            out.append(json.loads(path.read_text()))
        else:
            out.append(None)
    return out


def _fold_stat(fold_jsons: List[Optional[Dict]], metric: str) -> Dict:
    """Compute {median, min, max, values} for a metric across folds."""
    vals = []
    for j in fold_jsons:
        if j is None:
            continue
        v = _resolve(j, metric)
        if v is None or not isinstance(v, (int, float)):
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        vals.append(float(v))
    if not vals:
        return {"median": None, "min": None, "max": None, "values": []}
    return {
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "values": vals,
    }


def _variant_row(variant: str, fold_jsons: List[Optional[Dict]],
                 metrics: List[str]) -> Dict:
    return {
        "variant": variant,
        "n_folds": sum(1 for j in fold_jsons if j is not None),
        "metrics": {m: _fold_stat(fold_jsons, m) for m in metrics},
    }


PERCENT_METRICS = {"exact_match"}


def _fmt_stat(stat: Dict, digits: int = 4, scale: float = 1.0,
              suffix: str = "") -> str:
    if stat["median"] is None:
        return "—"
    fmt = f"{{:.{digits}f}}"
    return (f"{fmt.format(stat['median'] * scale)}{suffix} "
            f"[{fmt.format(stat['min'] * scale)}{suffix}, "
            f"{fmt.format(stat['max'] * scale)}{suffix}]")


def _print_band(title: str, rows: List[Dict], metrics: List[str],
                digits: int = 4):
    print(f"\n### {title}")
    hdr = ["variant (n)"] + metrics
    widths = [max(len(hdr[0]),
                  *(len(f"{r['variant']} ({r['n_folds']})") for r in rows))] + \
             [max(len(m), 22) for m in metrics]
    line = " | ".join(h.ljust(w) for h, w in zip(hdr, widths))
    print(line)
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        cells = [f"{r['variant']} ({r['n_folds']})".ljust(widths[0])]
        for m, w in zip(metrics, widths[1:]):
            if m in PERCENT_METRICS:
                cells.append(_fmt_stat(r["metrics"][m], digits=2,
                                       scale=100.0, suffix="%").ljust(w))
            else:
                cells.append(_fmt_stat(r["metrics"][m], digits).ljust(w))
        print(" | ".join(cells))


PER_STATE_METRICS = ["ned_mean", "exact_match", "vendi_ratio_gen_over_ref"]


def _extract_per_state(obj: Dict) -> Optional[Dict[str, Dict]]:
    """Pull `per_state` from a fold JSON, trying mode prefixes."""
    for probe in ("sampled_per_state", "baseline_per_state",
                  "greedy_per_state", "per_state"):
        if probe in obj and isinstance(obj[probe], dict):
            return obj[probe]
    return None


def _print_per_state_metrics(model_variants, floor_variants,
                             pat_lookup, result_dir: Path, split: str):
    """Per-FSM-state NED / exact match / Vendi ratio, aggregated across folds.

    For each metric, one sub-band with columns = states (median [min, max]
    across folds + avg bucket size). Pulled from the nested `per_state` dict
    in each fold JSON, so _fold_stat can't handle it.
    """
    from collections import defaultdict

    def _collect(variants):
        per_variant = {}
        for name in variants:
            fold_js = _load_variant(name, pat_lookup[name], result_dir, split)
            # {metric: {state: [values]}}, {state: [ns]}
            by_metric = defaultdict(lambda: defaultdict(list))
            by_state_n = defaultdict(list)
            for j in fold_js:
                if j is None:
                    continue
                per_state = _extract_per_state(j)
                if per_state is None:
                    continue
                for state, d in per_state.items():
                    by_state_n[state].append(d.get("n", 0))
                    for m in PER_STATE_METRICS:
                        if m in d:
                            by_metric[m][state].append(d[m])
            per_variant[name] = (by_metric, by_state_n)
        return per_variant

    def _render(metric: str, title: str, per_variant):
        all_states = sorted(
            {s for (by_m, _) in per_variant.values() for s in by_m.get(metric, {})},
            key=lambda x: int(x),
        )
        if not all_states:
            return
        print(f"\n### {title} — {metric}")
        hdr = ["variant"] + [f"state {s} (avg n)" for s in all_states]
        widths = [max(12, *(len(v) for v in per_variant))] + [26] * len(all_states)
        print(" | ".join(h.ljust(w) for h, w in zip(hdr, widths)))
        print("-+-".join("-" * w for w in widths))
        for name, (by_m, by_n) in per_variant.items():
            cells = [name.ljust(widths[0])]
            m_map = by_m.get(metric, {})
            is_pct = metric in PERCENT_METRICS
            scale = 100.0 if is_pct else 1.0
            sfx = "%" if is_pct else ""
            digits = 2 if is_pct else 4
            for s, w in zip(all_states, widths[1:]):
                vals = m_map.get(s, [])
                ns = by_n.get(s, [])
                if vals:
                    med = float(np.median(vals)) * scale
                    lo = float(np.min(vals)) * scale
                    hi = float(np.max(vals)) * scale
                    avg_n = int(np.mean(ns)) if ns else 0
                    cells.append(
                        f"{med:.{digits}f}{sfx} [{lo:.{digits}f}{sfx},"
                        f"{hi:.{digits}f}{sfx}] ({avg_n})".ljust(w)
                    )
                else:
                    cells.append("—".ljust(w))
            print(" | ".join(cells))

    model_data = _collect(model_variants)
    floor_data = _collect(floor_variants)
    for metric in PER_STATE_METRICS:
        _render(metric, "Per-state — Model", model_data)
        _render(metric, "Per-state — Floor", floor_data)


def _floor_ceiling_flag(model_rows: List[Dict], floor_rows: List[Dict],
                        metric: str) -> Optional[str]:
    """Flag a metric as non-discriminative if best floor median ≥ 0.95 × best model median.

    For metrics where lower is better (ned_mean, self_bleu_generated) the ratio
    flips: flag if worst floor ≤ 1.05 × best model.
    """
    model_medians = [r["metrics"][metric]["median"] for r in model_rows
                     if r["metrics"][metric]["median"] is not None]
    floor_medians = [r["metrics"][metric]["median"] for r in floor_rows
                     if r["metrics"][metric]["median"] is not None]
    if not model_medians or not floor_medians:
        return None

    lower_is_better = metric in ("ned_mean", "ned_median", "self_bleu_generated",
                                 "token_kl_divergence")
    if lower_is_better:
        best_model = min(model_medians)
        best_floor = min(floor_medians)
        if best_floor <= 1.05 * best_model:
            return f"floor {best_floor:.4f} ≤ 1.05×model {best_model:.4f}"
    else:
        best_model = max(model_medians)
        best_floor = max(floor_medians)
        if best_floor >= 0.95 * best_model:
            return f"floor {best_floor:.4f} ≥ 0.95×model {best_model:.4f}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default=None,
                    help="Optional JSON path to write the aggregated results")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    split = args.split

    variants = [
        ("MDLM",                 "mdlm_fold_{fold}/mdlm_eval_{split}.json"),
        ("AR-modern",            "ar_modern_fold_{fold}/ar_modern_eval_{split}.json"),
        ("BL-3 (Chronos+VJEPA)", "bl3_fold_{fold}/bl3_eval_{split}.json"),
        ("BL-4 (Chronos+CoTrk)", "bl4_fold_{fold}/bl4_eval_{split}.json"),
        ("BL-6 (MOMENT)",        "bl6_fold_{fold}/bl6_eval_{split}.json"),
    ]
    baselines = [
        ("unigram",       "baselines_fold_{fold}/unigram_eval_{split}.json"),
        ("copy_modal",    "baselines_fold_{fold}/copy_modal_eval_{split}.json"),
        ("copy_previous", "baselines_fold_{fold}/copy_previous_eval_{split}.json"),
    ]

    all_metrics = HEADLINE_METRICS + DIAGNOSTIC_METRICS

    model_rows = []
    for name, pat in variants:
        fold_js = _load_variant(name, pat, result_dir, split)
        row = _variant_row(name, fold_js, all_metrics)
        # Timing only makes sense for model variants.
        row["timing"] = {m: _fold_stat(fold_js, m) for m in TIMING_METRICS}
        model_rows.append(row)

    floor_rows = []
    for name, pat in baselines:
        fold_js = _load_variant(name, pat, result_dir, split)
        floor_rows.append(_variant_row(name, fold_js, all_metrics))

    print(f"\n==============================")
    print(f"5-fold aggregation (split={split})")
    print(f"==============================")

    _print_band("Headline metrics — Model", model_rows, HEADLINE_METRICS)
    _print_band("Headline metrics — Floor", floor_rows, HEADLINE_METRICS)

    _print_band("Diagnostic metrics — Model", model_rows, DIAGNOSTIC_METRICS)

    _print_per_state_metrics(
        [v[0] for v in variants], [b[0] for b in baselines],
        pat_lookup={**dict(variants), **dict(baselines)},
        result_dir=result_dir, split=split,
    )

    print("\n### Timing (model variants)")
    timing_hdr = ["variant"] + TIMING_METRICS
    w0 = max(len("variant"), *(len(r["variant"]) for r in model_rows))
    widths = [w0] + [24] * len(TIMING_METRICS)
    print(" | ".join(h.ljust(w) for h, w in zip(timing_hdr, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in model_rows:
        cells = [r["variant"].ljust(widths[0])]
        for m, w in zip(TIMING_METRICS, widths[1:]):
            cells.append(_fmt_stat(r["timing"][m], digits=4).ljust(w))
        print(" | ".join(cells))

    print("\n### Discriminability flags")
    any_flag = False
    for m in HEADLINE_METRICS:
        flag = _floor_ceiling_flag(model_rows, floor_rows, m)
        if flag is not None:
            print(f"  ⚠ {m}: {flag}")
            any_flag = True
    if not any_flag:
        print("  (none — every headline metric separates models from the floor)")

    if args.out:
        payload = {
            "split": split,
            "models": model_rows,
            "floor": floor_rows,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nSaved aggregated JSON -> {args.out}")


if __name__ == "__main__":
    main()
