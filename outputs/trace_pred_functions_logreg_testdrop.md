# Trace Prediction Report

## Setup

- Runs: `1`
- Target: `functions`
- Features: `fused`
- Model: `logreg`
- Missing mode: `test_drop`
- Window size: `10 ms`

## Dataset Stats

- Windows: `3452`
- Features: `20`
- Label cardinality (avg positives/window): `40.374`
- Label space (total/used): `64/50`

Top labels:
- `__towrite(91)`: 3451
- `pop_arg(103)`: 3451
- `waxi_dlr_user_end_access(5)`: 3451
- `waxi_dlr_user_read_bytes(4)`: 3451
- `waxi_sdk_log(45)`: 3451
- `waxi_dlr_user_write_bytes(6)`: 3451
- `sn_write(107)`: 3451
- `vfprintf(101)`: 3451
- `waxi_logging_log(7)`: 3451
- `waxi_dlr_user_begin_access(3)`: 3451

## Metrics

| Metric | Value |
|---|---:|
| Micro-F1 | 0.9710 |
| Macro-F1 | 0.8783 |
| Precision@5 | 0.9962 |
| Precision@10 | 0.9975 |
| Micro-F1 (missing syslog) | 0.1304 |
| Micro-F1 (missing datalayer) | 0.9659 |

## Interpretability Summary

- Removing syslog changes Micro-F1 by `+0.8406`.
- Removing datalayer changes Micro-F1 by `+0.0050`.

Top-function per-class F1 (top-10 by frequency):
- `__towrite(91)`: F1=1.0000, support=691
- `waxi_dlr_user_begin_access(3)`: F1=1.0000, support=691
- `__fwritex(92)`: F1=1.0000, support=691
- `waxi_end_access(27)`: F1=1.0000, support=691
- `printf(80)`: F1=1.0000, support=691
- `printf_core(102)`: F1=1.0000, support=691
- `memset(109)`: F1=1.0000, support=691
- `memcpy(108)`: F1=1.0000, support=691
- `execute(21)`: F1=1.0000, support=691
- `ethercat_write(29)`: F1=1.0000, support=691