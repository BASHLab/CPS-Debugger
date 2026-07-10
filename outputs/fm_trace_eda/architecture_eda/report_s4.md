# Architecture-Informing EDA — CPS-Debugger FM Trace Generation
# 2026-04-09 00:55:59
# sections=4
================================================================================
Usable sessions: 29, Ctrl+math edges: 318, Variable: 32

>>> Running section 4

================================================================================
# Section 4: Temporal Autocorrelation of Variable Edges
================================================================================

  Loading 20000 contiguous ticks from 2025-03-18_12-39-10 (rows 242587–262587)
  Got 20000 ticks

                Edge             Function    lag=1   lag=2   lag=5  lag=10  lag=50 lag=100 lag=500lag=1000
----------------------------------------------------------------------------------------------------
          31:101:169      waxi_end_access   -0.000  -0.000  -0.000  -0.000  -0.000  -0.000  -0.000  -0.000
            33:19:21             pou_main      N/A     N/A     N/A     N/A     N/A     N/A     N/A     N/A
            33:19:46             pou_main      N/A     N/A     N/A     N/A     N/A     N/A     N/A     N/A
            33:24:46             pou_main      N/A     N/A     N/A     N/A     N/A     N/A     N/A     N/A
            33:52:54             pou_main      N/A     N/A     N/A     N/A     N/A     N/A     N/A     N/A
           33:52:102             pou_main      N/A     N/A     N/A     N/A     N/A     N/A     N/A     N/A
            33:57:59             pou_main      N/A     N/A     N/A     N/A     N/A     N/A     N/A     N/A
           33:57:102             pou_main      N/A     N/A     N/A     N/A     N/A     N/A     N/A     N/A
            33:71:73             pou_main      N/A     N/A     N/A     N/A     N/A     N/A     N/A     N/A
          33:112:114             pou_main      N/A     N/A     N/A     N/A     N/A     N/A     N/A     N/A
          33:112:129             pou_main      N/A     N/A     N/A     N/A     N/A     N/A     N/A     N/A
          33:116:118             pou_main      N/A     N/A     N/A     N/A     N/A     N/A     N/A     N/A
             34:8:10   turn_on_with_delay      N/A     N/A     N/A     N/A     N/A     N/A     N/A     N/A
            34:19:21   turn_on_with_delay   -0.007  -0.005   0.012  -0.002   0.004  -0.004  -0.006  -0.001
            34:19:46   turn_on_with_delay   -0.007  -0.005   0.012  -0.002   0.004  -0.004  -0.006  -0.001
            34:24:46   turn_on_with_delay   -0.012  -0.003   0.024  -0.008  -0.000   0.004  -0.016  -0.001
            34:52:54   turn_on_with_delay   -0.007  -0.005   0.012  -0.002   0.004  -0.004  -0.006  -0.001
           34:52:102   turn_on_with_delay   -0.012  -0.003   0.024  -0.008  -0.000   0.004  -0.016  -0.001
            34:57:59   turn_on_with_delay   -0.020  -0.011   0.009   0.006  -0.002   0.005  -0.008  -0.002
           34:57:102   turn_on_with_delay    0.002  -0.004   0.018   0.006  -0.012   0.004   0.012   0.011
            34:71:73   turn_on_with_delay   -0.005  -0.011  -0.000   0.012  -0.004  -0.001  -0.001   0.003
            34:71:88   turn_on_with_delay   -0.011   0.003   0.003  -0.010   0.006  -0.001  -0.012  -0.001
          34:112:114   turn_on_with_delay    0.002  -0.004   0.018   0.006  -0.012   0.004   0.012   0.011
          34:112:129   turn_on_with_delay   -0.012  -0.003   0.024  -0.008  -0.000   0.004  -0.016  -0.001
          34:116:118   turn_on_with_delay    0.002  -0.004   0.018   0.006  -0.012   0.004   0.012   0.011
             63:8:10                 fmin    0.009   0.008  -0.001  -0.004  -0.005  -0.004   0.000  -0.001
            63:17:22                 fmin    0.009   0.008  -0.001  -0.004  -0.005  -0.004   0.000  -0.001
             64:8:10                 fmax    0.009   0.008  -0.001  -0.004  -0.005  -0.004   0.000  -0.001
            64:17:22                 fmax    0.009   0.008  -0.001  -0.004  -0.005  -0.004   0.000  -0.001
            77:29:31                  log    0.001  -0.006  -0.010   0.005   0.001   0.003   0.003  -0.010
           77:29:252                  log   -0.004   0.005  -0.003   0.006  -0.009  -0.005  -0.010  -0.004
          77:275:388                  log   -0.004   0.005  -0.003   0.006  -0.009  -0.005  -0.010  -0.004

  lag=1: mean=-0.003, std=0.008, min=-0.021, max=0.009

  lag=10: mean=-0.000, std=0.006, min=-0.010, max=0.012

  lag=100: mean=-0.000, std=0.004, min=-0.005, max=0.005

  lag=1000: mean=0.000, std=0.005, min=-0.010, max=0.011

  Identical consecutive ticks: 0.1317 (13.2%)
  Edges changing per tick: mean=6.58, std=3.72, max=19
  Cosine similarity: mean=0.9033, std=0.0793

**Implication**: Only 13% identical — low persistence. Need strong sensor conditioning per tick.

[Section 4: 13.1s]


Runtime: 0.2 minutes