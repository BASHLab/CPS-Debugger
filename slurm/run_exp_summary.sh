#!/bin/bash
#SBATCH --job-name=cps_exp_summary
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.out
#SBATCH --error=/home/simran/allspark-data-exploration/CPS-Debugger/outputs/slurm/%x_%j.err

# Prints a summary table of all experiment results after exp01 and exp06 complete.

set -euo pipefail

echo "Host: $(hostname)"
echo "Start: $(date)"

PYTHON=/home/simran/.conda/envs/slimllm/bin/python3
REPO=/home/simran/allspark-data-exploration/CPS-Debugger
cd "$REPO"

$PYTHON -u - <<'PYEOF'
import json
from pathlib import Path

OUT = Path("outputs/experiments")

def load(name):
    p = OUT / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

print("\n" + "="*60)
print("CPS EXPERIMENT SUMMARY")
print("="*60)

# Exp 01: branch count regression
r = load("branch_regression_results.json")
if r:
    print(f"\n[01] Branch Count Regression (within-run temporal CV, n={r.get('n_windows','?')} windows)")
    print(f"     All features:   mean R²={r['all_features']['mean_r2']:.3f}, "
          f"frac R²>0.5={r['all_features']['frac_r2_above_0_5']:.0%}")
    print(f"     Physical only:  mean R²={r['physical_only']['mean_r2']:.3f}, "
          f"frac R²>0.3={r['physical_only']['frac_r2_above_0_3']:.0%}")
    print(f"     Syslog only:    mean R²={r['syslog_only']['mean_r2']:.3f}")
    d = r.get("decomposition", {})
    print(f"     Decomposition:  {d.get('physical_unique',0)} phy-unique, "
          f"{d.get('syslog_unique',0)} syslog-unique, "
          f"{d.get('both',0)} both, {d.get('neither',0)} neither")

# Exp 02: setpoint classification
r2 = load("setpoint_results.json")
if r2:
    print(f"\n[02] Within-BALANCE Setpoint Classification (4 classes, n={r2.get('n_windows','?')} windows)")
    for fs_name, res in r2.get("results", {}).items():
        print(f"     {fs_name:22s}: acc={res['accuracy']:.3f}, macro_F1={res['macro_f1']:.3f}")

# Exp 03: anomaly detection
r3 = load("anomaly_detection_results.json")
if r3:
    print(f"\n[03] Anomaly Detection (synthetic injection, n_test={r3.get('n_test','?')})")
    for name, auc in r3.get("auc_scores", {}).items():
        print(f"     {name:45s}: AUC={auc:.3f}")

# Exp 05: audio trace
r5 = load("audio_trace_results.json")
if r5:
    print(f"\n[05] Audio → Trace (Feb04 run, n={r5.get('n_windows','?')} windows)")
    print(f"     Audio only:       mean R²={r5['audio_only']['mean_r2']:.3f}, "
          f"frac R²>0.3={r5['audio_only']['frac_r2_above_0_3']:.0%}")
    print(f"     Audio + Syslog:   mean R²={r5['audio_syslog']['mean_r2']:.3f}, "
          f"frac R²>0.3={r5['audio_syslog']['frac_r2_above_0_3']:.0%}")

# Exp 06: cross-run generalization
r6 = load("cross_run_results.json")
if r6:
    print(f"\n[06] Cross-Run Generalization (leave-one-run-out, {len(r6.get('per_run',[]))} folds)")
    runs = r6.get("per_run", [])
    if runs:
        for row in runs:
            print(f"     Test={row['test_run']}: R²={row['mean_r2']:.3f} "
                  f"(phy={row.get('r2_physical_only','?'):.3f}), "
                  f"microF1={row['micro_f1']:.3f}")
    agg = r6.get("aggregate", {})
    print(f"     Mean across runs: R²={agg.get('mean_r2_mean','?'):.3f}±{agg.get('mean_r2_std','?'):.3f}, "
          f"microF1={agg.get('micro_f1_mean','?'):.3f}±{agg.get('micro_f1_std','?'):.3f}")

print("\n" + "="*60)
PYEOF

echo "Done: $(date)"
