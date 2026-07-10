# Architecture-Informing EDA — CPS-Debugger FM Trace Generation
# 2026-04-09 00:55:59
# sections=9
================================================================================
Usable sessions: 29, Ctrl+math edges: 318, Variable: 32

>>> Running section 9

================================================================================
# Section 9: Sequence Length Distribution
================================================================================

  Note: processed_trace gives counts, not ordered sequences.
  'Sequence length' = total branch count per tick.
  Scanning 2025-03-13_09-23-44...
  Scanning 2025-03-13_11-12-19...
  Scanning 2025-03-13_13-31-50...
  Scanning 2025-03-13_14-32-27...
  Scanning 2025-03-17_10-06-14...

  All edges:
    min=3, max=1904, mean=1875.7, median=1879, std=30.5
    p5=1864, p25=1876, p75=1881, p95=1890, p99=1894

  Controller+math:
    min=3, max=101, mean=90.7, median=91, std=4.2
    p5=87, p25=89, p75=92, p95=98, p99=100

**Implication**: Ctrl+math ~91 tokens/tick — very manageable for AR generation.

[Section 9: 29.0s]


Runtime: 0.5 minutes