# Trace Prediction Report

## Setup

- Runs: `1`
- Target: `edges`
- Features: `fused`
- Model: `logreg`
- Missing mode: `test_drop`
- Window size: `10 ms`

## Dataset Stats

- Windows: `3452`
- Features: `20`
- Label cardinality (avg positives/window): `51.673`
- Label space (total/used): `93/67`

Top labels:
- `ethercat_write(29)->waxi_dlr_user_read_bytes(4)`: 3451
- `waxi_end_access(27)->waxi_dlr_user_end_access(5)`: 3451
- `vfprintf(101)->sn_write(107)`: 3451
- `vsnprintf(106)->vfprintf(101)`: 3451
- `ethercat_write(29)->waxi_dlr_user_begin_access(3)`: 3451
- `printf_core(102)->__fwritex(92)`: 3451
- `vfprintf(101)->printf_core(102)`: 3451
- `waxi_sdk_log(45)->memset(109)`: 3451
- `sn_write(107)->memcpy(108)`: 3451
- `waxi_sdk_log(45)->waxi_logging_log(7)`: 3451

## Metrics

| Metric | Value |
|---|---:|
| Micro-F1 | 0.9774 |
| Macro-F1 | 0.8975 |
| Precision@5 | 0.9983 |
| Precision@10 | 0.9984 |
| Micro-F1 (missing syslog) | 0.1319 |
| Micro-F1 (missing datalayer) | 0.9731 |

## Interpretability Summary

- Removing syslog changes Micro-F1 by `+0.8454`.
- Removing datalayer changes Micro-F1 by `+0.0043`.