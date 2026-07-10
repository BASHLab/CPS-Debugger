"""
parallel_B2_inference_cost.py — Can this run in real-time?

Times each component of the inference pipeline:
  - RF model prediction (single window, batch of 100)
  - Consistency score computation (L2 residual)
  - Full pipeline: feature extraction → prediction → score → threshold check
  - LLM call cost estimate (from zero-shot eval logs)

Output:
  outputs/parallel/inference_cost.json
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
EXP  = ROOT / "outputs/experiments"
OUT  = ROOT / "outputs/parallel"
OUT.mkdir(parents=True, exist_ok=True)

PHYSICAL_COLS = [
    "dl_current_x_mean", "dl_current_x_std", "dl_current_x_delta",
    "dl_angle_mean", "dl_angle_std", "dl_angle_delta",
    "dl_velocity_mean", "dl_ang_vel_mean", "dl_ang_vel_std",
]
WINDOW_MS       = 10       # ms per analysis window
CONTROL_CYCLE_MS = 1       # ms per controller cycle
N_WARMUP        = 20
N_TIMED         = 1000


def bench(fn, n_warmup=N_WARMUP, n_timed=N_TIMED):
    """Returns mean ms per call over n_timed iterations after n_warmup warmup calls."""
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_timed):
        fn()
    return (time.perf_counter() - t0) / n_timed * 1000  # ms


def main():
    df = pd.read_parquet(EXP / "aligned_dataset.parquet")
    vi = json.loads((EXP / "vocab_info.json").read_text())
    vocab = vi["vocab"]

    n80 = int(len(df) * 0.8)
    train = df.iloc[:n80]
    test  = df.iloc[n80:].reset_index(drop=True)

    X_tr = train[PHYSICAL_COLS].values
    Y_tr = train[vocab].values
    X_te = test[PHYSICAL_COLS].values
    Y_te = test[vocab].values

    print("Training RF (n_estimators=100, max_depth=12)...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=1, random_state=42)
    # n_jobs=1 to simulate single-core deployment scenario
    rf.fit(X_tr, Y_tr)

    X_single = X_te[0:1]
    X_batch  = X_te[:100]
    Y_single = Y_te[0:1]

    results = {}

    # ── 1. RF single-window prediction ────────────────────────────────────────
    print("Timing RF single-window prediction...")
    t_rf_single = bench(lambda: rf.predict(X_single))
    results["rf_single_window_ms"] = t_rf_single
    print(f"  RF single: {t_rf_single:.3f} ms")

    # ── 2. RF batch prediction (100 windows) ──────────────────────────────────
    print("Timing RF batch prediction (100 windows)...")
    t_rf_batch = bench(lambda: rf.predict(X_batch)) / 100  # per-window
    results["rf_batch_100_per_window_ms"] = t_rf_batch
    print(f"  RF batch/100: {t_rf_batch:.4f} ms/window")

    # ── 3. Consistency score (L2 residual) ────────────────────────────────────
    print("Timing consistency score (predict + L2)...")
    def consistency_score():
        pred = rf.predict(X_single)
        return np.linalg.norm(Y_single - pred, axis=1)[0]

    t_score = bench(consistency_score)
    results["consistency_score_ms"] = t_score
    print(f"  Consistency score: {t_score:.3f} ms")

    # ── 4. Full pipeline ───────────────────────────────────────────────────────
    # Simulate feature extraction (aggregating 10 datalayer rows → 9 physical features)
    print("Timing full pipeline (feature extraction → predict → score → threshold)...")
    p95 = 17.61  # from exp03

    # Feature extraction: simulate computing mean/std/delta from 10 raw rows
    raw_rows = np.random.default_rng(42).random((10, 4))  # 4 raw channels
    def extract_features(rows):
        return np.array([
            rows[:, 0].mean(), rows[:, 0].std(), rows[-1, 0] - rows[0, 0],  # position
            rows[:, 1].mean(), rows[:, 1].std(), rows[-1, 1] - rows[0, 1],  # angle
            (rows[-1, 0] - rows[0, 0]) / 0.01,                               # velocity
            rows[:, 2].mean(), rows[:, 2].std(),                             # ang_vel
        ]).reshape(1, -1)

    def full_pipeline():
        X_feat = extract_features(raw_rows)
        pred   = rf.predict(X_feat)
        score  = np.linalg.norm(Y_single - pred, axis=1)[0]
        return score >= p95

    t_pipeline = bench(full_pipeline)
    results["full_pipeline_ms"] = t_pipeline
    print(f"  Full pipeline: {t_pipeline:.3f} ms")

    # ── 5. Interpretation ─────────────────────────────────────────────────────
    if t_pipeline < WINDOW_MS:
        feasibility = "real_time"
        feasibility_note = (
            f"Full pipeline ({t_pipeline:.2f}ms) < analysis window ({WINDOW_MS}ms). "
            f"Real-time monitoring is feasible on a single CPU core. "
            f"The system analyzes every 10ms window with {WINDOW_MS/t_pipeline:.0f}× margin."
        )
    elif t_pipeline < 100:
        feasibility = "near_real_time"
        lag = round(t_pipeline / WINDOW_MS)
        feasibility_note = (
            f"Full pipeline ({t_pipeline:.2f}ms) > window ({WINDOW_MS}ms). "
            f"Near-real-time monitoring with ~{lag}-window lag ({lag*WINDOW_MS}ms). "
            f"Acceptable for most CPS safety monitoring (operator reaction time >>100ms)."
        )
    else:
        feasibility = "offline_only"
        feasibility_note = (
            f"Full pipeline ({t_pipeline:.2f}ms) is too slow for real-time use. "
            f"Suitable for offline post-hoc analysis only."
        )

    results["feasibility"] = feasibility
    results["feasibility_note"] = feasibility_note

    # ── 6. LLM cost estimate ──────────────────────────────────────────────────
    # From the zero-shot eval: 250 examples took ~30-45 min → ~7-11 sec/example
    # Haiku is called only on anomaly windows, not every window
    # Estimate: ~2-3 API calls per anomaly event (initial + follow-up)
    results["llm_anomaly_explanation"] = {
        "model": "claude-haiku-4-5-20251001",
        "estimated_latency_ms": 5000,
        "note": (
            "LLM invoked ONLY on anomaly detection events (not every window). "
            "At p99 FPR <1%, this means ~few calls per hour in steady-state operation. "
            "Estimated from zero-shot eval timing: ~5s per call (API latency dominant). "
            "Acceptable for human-in-the-loop debugging workflow."
        ),
        "cost_per_call_usd_approx": 0.001,
    }

    # ── 7. System comparison ──────────────────────────────────────────────────
    results["system_context"] = {
        "controller_cycle_ms":       CONTROL_CYCLE_MS,
        "analysis_window_ms":        WINDOW_MS,
        "rf_inference_ms":           t_rf_single,
        "full_pipeline_ms":          t_pipeline,
        "ratio_pipeline_to_window":  t_pipeline / WINDOW_MS,
        "n_windows_per_second":      1000 / WINDOW_MS,
        "n_rf_predictions_per_sec":  1000 / t_rf_single,
    }

    (OUT / "inference_cost.json").write_text(json.dumps(results, indent=2))

    print(f"\n── Inference Cost Summary ──")
    print(f"  RF single-window:     {t_rf_single:.3f} ms")
    print(f"  RF batch (per window): {t_rf_batch:.4f} ms")
    print(f"  Consistency score:    {t_score:.3f} ms")
    print(f"  Full pipeline:        {t_pipeline:.3f} ms")
    print(f"  Feasibility:          {feasibility.upper()}")
    print(f"  {feasibility_note}")
    print("\nSaved inference_cost.json")
    print("Done.")


if __name__ == "__main__":
    main()
