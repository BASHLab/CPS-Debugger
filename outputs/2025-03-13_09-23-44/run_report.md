# CPS Debugger Report

## Task

Build a lightweight CPS debugger baseline that predicts execution-level behavior from synchronized
telemetry, then highlights time windows where observed execution is unlikely under the learned model.

## Why This Matters

- Cyber-physical incidents often appear first as cross-modal inconsistency (physical state looks normal, but control flow changes).
- Full trace inspection at every cycle is expensive; a model can triage suspicious windows quickly.
- Missing modalities are common in production capture pipelines, so robustness to absent inputs matters.

## Run Context

- Run path: `/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted/2025-03-13_09-23-44`
- Approximate run span: `2025-03-13 13:23:51.330000 UTC` to `2025-03-13 13:24:26.040000 UTC`
- Window size: `10000` us (`10.0` ms)
- Label mode: `triplets`
- Vocabulary source: `py_literal`
- Vocabulary file: `/home/simran/allspark-data-exploration/eda/triplet_to_idx.json`

## Method Summary

- Aggregate `datalayer` and `syslog` signals per time window into statistical/dynamic features.
- Convert trace data into multi-label execution targets (`triplets` from `cf_table` here).
- Train a one-vs-rest logistic regression baseline to map sensor features -> execution labels.
- Score each window with negative log-likelihood; larger score means more surprising execution.
- Localize anomalies by listing labels with the highest surprise within each top window.

## Quantitative Results

- Micro-F1: `0.9704`
- Macro-F1: `0.7872`
- Micro-F1 (missing syslog): `0.8052`
- Robustness note: removing syslog reduces Micro-F1 by `0.1652` (`17.0%` relative).
- Windows: `3452`
- Features: `14`
- Labels (used / total): `339 / 407`
- Train/Test windows: `2761 / 691`

## Interpretation

- High Micro-F1 indicates strong performance on frequent control-flow patterns.
- Lower Macro-F1 indicates rarer labels are harder and remain a key improvement area.
- The syslog ablation gap quantifies dependence on system-health context for prediction stability.

## Top Anomalous Windows

These are ranked by total negative log-likelihood. Each row lists the most surprising localized labels.

### Rank 1 | win=174187223154 | ts=2025-03-13 13:23:51.540000 UTC | score=1222.581
- `pou_general_machine(37):43->67` (surprise=13.816)
- `turn_off_with_delay(35):20->46` (surprise=13.816)
- `turn_off_with_delay(35):51->53` (surprise=13.816)
- `turn_off_with_delay(35):69->71` (surprise=13.816)
- `turn_on_with_delay(34):24->26` (surprise=13.816)

### Rank 2 | win=174187223134 | ts=2025-03-13 13:23:51.340000 UTC | score=701.161
- `waxi_end_access(27):101->169` (surprise=4.994)
- `memset(109):8->10` (surprise=4.994)
- `memset(109):8->22` (surprise=4.994)
- `memset(109):28->30` (surprise=4.994)
- `memset(109):57->384` (surprise=4.994)

### Rank 3 | win=174187224018 | ts=2025-03-13 13:24:00.180000 UTC | score=187.409
- `tick(22):1209->1211` (surprise=9.343)
- `tick(22):1634->2765` (surprise=9.343)
- `tick(22):1375->1377` (surprise=7.299)
- `tick(22):1397->1632` (surprise=7.299)
- `tick(22):1287->1311` (surprise=7.210)

### Rank 4 | win=174187224597 | ts=2025-03-13 13:24:05.970000 UTC | score=185.036
- `tick(22):1209->1211` (surprise=13.816)
- `tick(22):1634->2765` (surprise=13.816)
- `tick(22):1287->1289` (surprise=13.816)
- `tick(22):1309->1633` (surprise=13.816)
- `turn_on_with_delay(34):71->73` (surprise=1.443)

### Rank 5 | win=174187223156 | ts=2025-03-13 13:23:51.560000 UTC | score=167.918
- `__rem_pio2(72):67->641` (surprise=2.877)
- `__rem_pio2(72):755->757` (surprise=2.877)
- `__rem_pio2(72):762->773` (surprise=2.877)
- `__rem_pio2(72):778->780` (surprise=2.877)
- `__rem_pio2(72):849->851` (surprise=2.877)

### Rank 6 | win=174187224938 | ts=2025-03-13 13:24:09.380000 UTC | score=160.852
- `pou_standup_rel(39):511->513` (surprise=7.409)
- `pou_standup_rel(39):526->528` (surprise=7.032)
- `tick(22):1375->1399` (surprise=6.847)
- `pou_standup_rel(39):456->458` (surprise=6.847)
- `pou_lqr_sim(40):44->46` (surprise=6.595)

### Rank 7 | win=174187223999 | ts=2025-03-13 13:23:59.990000 UTC | score=141.682
- `tick(22):1676->1806` (surprise=9.054)
- `tick(22):1889->1930` (surprise=9.054)
- `tick(22):2093->2095` (surprise=6.615)
- `tick(22):2173->2763` (surprise=6.615)
- `stays_balanced(20):58->60` (surprise=6.615)

### Rank 8 | win=174187223998 | ts=2025-03-13 13:23:59.980000 UTC | score=135.603
- `tick(22):1676->1806` (surprise=8.498)
- `tick(22):1889->1930` (surprise=8.498)
- `tick(22):2093->2095` (surprise=6.826)
- `tick(22):2173->2763` (surprise=6.826)
- `stays_balanced(20):58->60` (surprise=6.826)

### Rank 9 | win=174187223155 | ts=2025-03-13 13:23:51.550000 UTC | score=133.802
- `turn_on_with_delay(34):71->73` (surprise=2.442)
- `pou_drive_control_word(42):84->86` (surprise=2.362)
- `pou_drive_control_word(42):88->90` (surprise=2.362)
- `pou_drive_control_word(42):93->95` (surprise=2.362)
- `pou_drive_control_word(42):115->140` (surprise=2.362)

### Rank 10 | win=174187224571 | ts=2025-03-13 13:24:05.710000 UTC | score=117.630
- `pou_standup_rel(39):526->542` (surprise=7.658)
- `pou_standup_rel(39):511->513` (surprise=7.149)
- `pou_standup_rel(39):526->528` (surprise=6.846)
- `pou_lqr_sim(40):44->46` (surprise=6.225)
- `pou_lqr_sim(40):83->185` (surprise=6.225)

### Rank 11 | win=174187224576 | ts=2025-03-13 13:24:05.760000 UTC | score=96.938
- `pou_lqr_sim(40):44->46` (surprise=5.029)
- `pou_lqr_sim(40):83->185` (surprise=5.029)
- `pou_lqr_sim(40):281->283` (surprise=5.029)
- `pou_lqr_sim(40):299->301` (surprise=5.029)
- `pou_lqr_sim(40):314->340` (surprise=5.029)

### Rank 12 | win=174187224000 | ts=2025-03-13 13:24:00.000000 UTC | score=92.757
- `turn_off_with_delay(35):69->86` (surprise=5.924)
- `tick(22):1676->1806` (surprise=4.733)
- `tick(22):1889->1930` (surprise=4.733)
- `tick(22):2093->2095` (surprise=4.029)
- `tick(22):2173->2763` (surprise=4.029)

## Artifacts

- Metrics JSON: `/home/simran/allspark-data-exploration/CPS-Debugger/outputs/2025-03-13_09-23-44/metrics.json`
- Anomaly table CSV: `/home/simran/allspark-data-exploration/CPS-Debugger/outputs/2025-03-13_09-23-44/anomalies.csv`
- This report: `/home/simran/allspark-data-exploration/CPS-Debugger/outputs/2025-03-13_09-23-44/run_report.md`

## Caveats

- This baseline is linear and window-local; it does not model longer temporal dependencies.
- Anomaly scores are relative to this run and split, not global calibrated probabilities.
- Stronger baselines could include sequence models, class balancing, and calibrated thresholds.