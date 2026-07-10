# Architecture-Informing EDA — CPS-Debugger FM Trace Generation
# 2026-04-09 00:55:59
# sections=8
================================================================================
Usable sessions: 29, Ctrl+math edges: 318, Variable: 32

>>> Running section 8

================================================================================
# Section 8: Cross-Session Consistency
================================================================================
  2025-03-13_09-23-44: 500 ticks sampled
  2025-03-13_11-12-19: 500 ticks sampled
  2025-03-13_13-31-50: 501 ticks sampled
  2025-03-13_14-32-27: 500 ticks sampled
  2025-03-17_10-06-14: 500 ticks sampled
  2025-03-17_10-21-19: 500 ticks sampled
  2025-03-17_10-36-44: 500 ticks sampled
  2025-03-17_10-51-58: 500 ticks sampled
  2025-03-17_11-06-39: 500 ticks sampled
  2025-03-18_12-39-10: 500 ticks sampled
  2025-03-19_10-05-47: 500 ticks sampled
  2025-03-19_10-20-35: 500 ticks sampled
  2025-03-19_10-37-56: 500 ticks sampled
  2025-03-19_11-10-13: 500 ticks sampled
  2025-03-20_09-31-56: 500 ticks sampled
  2025-03-20_09-46-30: 500 ticks sampled
  2025-03-20_10-03-00: 500 ticks sampled
  2025-03-20_10-18-01: 500 ticks sampled
  2025-03-20_10-37-02: 500 ticks sampled
  2025-03-21_10-22-23: 500 ticks sampled
  2025-03-21_10-38-47: 500 ticks sampled
  2025-03-21_10-59-39: 500 ticks sampled
  2025-03-21_11-21-42: 500 ticks sampled
  2025-03-25_13-23-42: 500 ticks sampled
  2025-03-25_13-39-06: 500 ticks sampled
  2025-03-26_10-12-02: 500 ticks sampled
  2025-03-26_10-27-20: 500 ticks sampled
  2025-03-26_10-48-00: 500 ticks sampled
  2025-03-26_11-03-04: 500 ticks sampled

Per-edge cross-session variance:
                  Edge             Function  Mean(means)  Std(means)      CV
---------------------------------------------------------------------------
            31:101:169      waxi_end_access       1.0000      0.0005  0.0005
              33:19:21             pou_main       0.0000      0.0000  0.0000
              33:19:46             pou_main       0.0000      0.0000  0.0000
              33:24:46             pou_main       0.0000      0.0000  0.0000
              33:52:54             pou_main       0.0000      0.0000  0.0000
             33:52:102             pou_main       0.0000      0.0000  0.0000
              33:57:59             pou_main       0.0000      0.0000  0.0000
             33:57:102             pou_main       0.0000      0.0000  0.0000
              33:71:73             pou_main       0.0000      0.0000  0.0000
            33:112:114             pou_main       0.0000      0.0000  0.0000
            33:112:129             pou_main       0.0000      0.0000  0.0000
            33:116:118             pou_main       0.0000      0.0000  0.0000
               34:8:10   turn_on_with_delay       2.9998      0.0011  0.0004
              34:19:21   turn_on_with_delay       1.7834      0.0523  0.0293
              34:19:46   turn_on_with_delay       1.2164      0.0522  0.0430
              34:24:46   turn_on_with_delay       1.4608      0.0653  0.0447
              34:52:54   turn_on_with_delay       1.2164      0.0522  0.0430
             34:52:102   turn_on_with_delay       1.4608      0.0653  0.0447
              34:57:59   turn_on_with_delay       0.9028      0.0524  0.0580
             34:57:102   turn_on_with_delay       0.3136      0.0470  0.1500
              34:71:73   turn_on_with_delay       0.5190      0.0796  0.1534
              34:71:88   turn_on_with_delay       0.3838      0.0707  0.1842
            34:112:114   turn_on_with_delay       0.3136      0.0470  0.1500
            34:112:129   turn_on_with_delay       1.4608      0.0653  0.0447
            34:116:118   turn_on_with_delay       0.3136      0.0470  0.1500
               63:8:10                 fmin       1.9115      0.1397  0.0731
              63:17:22                 fmin       1.9115      0.1397  0.0731
               64:8:10                 fmax       1.9115      0.1397  0.0731
              64:17:22                 fmax       1.9115      0.1397  0.0731
              77:29:31                  log       0.0199      0.0293  1.4691
             77:29:252                  log       0.0652      0.0833  1.2787
            77:275:388                  log       0.0652      0.0833  1.2787

  Mean CV across edges: 0.1692

  Sessions with highest deviation:
    2025-03-13_09-23-44: dev=0.2688
    2025-03-19_10-05-47: dev=0.0746
    2025-03-25_13-23-42: dev=0.0664
    2025-03-20_09-46-30: dev=0.0605
    2025-03-26_10-12-02: dev=0.0511

**Implication**: Moderate session effects (CV=0.169). Use session-level splits.

[Section 8: 292.9s]


Runtime: 4.9 minutes