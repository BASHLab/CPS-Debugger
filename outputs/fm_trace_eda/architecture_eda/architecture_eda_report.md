# Architecture-Informing EDA — CPS-Debugger FM Trace Generation
# 2026-04-09 00:55:59
# sections=1,2,6,10
================================================================================
Usable sessions: 29, Ctrl+math edges: 318, Variable: 32

>>> Running section 1

================================================================================
# Section 1: Sensor Feature Names, Types, and Statistics
================================================================================

Session: 2025-03-18_12-39-10  (572563 rows)
Columns: ['timestamp', 'pendulum_state', 'iteration', 'target_x', 'current_x', 'velocity', 'current_angle', 'angular_velocity', 'ts_us']

                 Col    dtype            min            max           mean            std   #uniq  %miss
---------------------------------------------------------------------------------------------------------
      pendulum_state    int64         0.0000         2.0000         0.9914         0.1798       3   0.0%
           iteration    int64         0.0000        35.0000        17.2293         9.9366      36   0.0%
            target_x  float64        -0.0900         0.0900         0.0113         0.0682       4   0.0%
           current_x  float64        -0.1701         0.1325         0.0086         0.0717  125057   0.0%
            velocity  float64        -1.0424         0.5897         0.0001         0.1071  255587   0.0%
       current_angle  float64        -3.1416         3.1408        -0.0000         0.3514    5410   0.0%
    angular_velocity  float64     -3141.5927       786.1652        -0.0549         9.1923     135   0.0%

Cross-session column check:
  2025-03-13_09-23-44: cols=['timestamp', 'pendulum_state', 'iteration', 'target_x', 'current_x', 'velocity', 'current_angle', 'angular_velocity'], states=[0]
  2025-03-13_11-12-19: cols=['timestamp', 'pendulum_state', 'iteration', 'target_x', 'current_x', 'velocity', 'current_angle', 'angular_velocity'], states=[0]
  2025-03-13_13-31-50: cols=['timestamp', 'pendulum_state', 'iteration', 'target_x', 'current_x', 'velocity', 'current_angle', 'angular_velocity'], states=[0]
  2025-03-13_14-32-27: cols=['timestamp', 'pendulum_state', 'iteration', 'target_x', 'current_x', 'velocity', 'current_angle', 'angular_velocity'], states=[0]
  2025-03-17_10-06-14: cols=['timestamp', 'pendulum_state', 'iteration', 'target_x', 'current_x', 'velocity', 'current_angle', 'angular_velocity'], states=[0]

**Implication**: 6 continuous sensors + 1 categorical state + 1 counter = 8 features. Simple linear projection suffices.

[Section 1: 0.4s]

>>> Running section 2

================================================================================
# Section 2: Trace Target Shape
================================================================================

  Tick 0: ts=1741872240295156
    Branches: 1867 total, 80 ctrl+math (75 edges), 1787 library (134 edges)
    Top-5 ctrl edges:
      34:8:10  count=3
      27:101:169  count=2
      34:19:46  count=2
      34:52:54  count=2
      21:101:113  count=1

  Tick 1: ts=1741872254035802
    Branches: 1880 total, 91 ctrl+math (80 edges), 1789 library (136 edges)

  Tick 2: ts=1741872232507805
    Branches: 1883 total, 96 ctrl+math (90 edges), 1787 library (134 edges)

  Tick 3: ts=1741872263907252
    Branches: 1877 total, 88 ctrl+math (76 edges), 1789 library (136 edges)

  Tick 4: ts=1741872251974699
    Branches: 1880 total, 88 ctrl+math (80 edges), 1792 library (135 edges)

  Ctrl+math count-value distribution (5 ticks):
    count=1: 365 occurrences
    count=2: 30 occurrences
    count=3: 6 occurrences

**Answer**: Each tick is a COUNT VECTOR — not binary, not ordered.
  cf_table gives aggregate counts per edge. Ordering lost in processed_trace.
  Only raw .aspk preserves execution order.
  Dimensionality: 318 ctrl+math edges. Most counts are 1; 32 edges vary (1-3).

**Implication**: This is multi-target low-cardinality count prediction (32-dim categorical), NOT long sequence generation — unless raw .aspk ordering is needed.

[Section 2: 0.0s]

>>> Running section 6

================================================================================
# Section 6: Graph Structure from layout.json
================================================================================

  Parsing /home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted/2025-03-13_09-23-44/code/layout.json

  Functions: 113
  Basic blocks: 1294
  Control-flow edges: 1981
  Instructions: 22231
  Avg insts/block: 17.2
  func_exports: {'_start': 19, 'execute': 21}
  import_func_count: 18

  Instruction types:
    type 0 (        branch):   1181 (5.3%)
    type 1 (   call_direct):    325 (1.5%)
    type 2 ( call_indirect):     13 (0.1%)
    type 3 (  block_marker):   1058 (4.8%)
    type 4 (        return):     71 (0.3%)
    type 5 (       regular):  19583 (88.1%)

  Top-10 functions by complexity:
    func 102               printf_core: 379 blk, 532 edg,  4528 inst, 83 calls
    func  49                  dlmalloc: 203 blk, 268 edg,  3021 inst,  7 calls
    func  71          __rem_pio2_large: 132 blk, 210 edg,  2215 inst,  4 calls
    func  22                      tick:  32 blk,  51 edg,  1334 inst, 18 calls
    func  30                  log_data:  22 blk,  25 edg,  1201 inst, 28 calls
    func  26             ethercat_read:  10 blk,  14 edg,   922 inst, 21 calls
    func  51                    dlfree:  60 blk,  80 edg,   796 inst,  0 calls
    func  29            ethercat_write:   9 blk,  12 edg,   743 inst, 14 calls
    func  21                   execute:  36 blk,  46 edg,   573 inst,  7 calls
    func 108                    memcpy:  32 blk,  50 edg,   515 inst,  0 calls

  Example: tick (func 22) — 32 blocks, first_block=209
    Block 209: 91 instructions
      pc=1, type=5, inst='global.get 0'
      pc=7, type=5, inst='local.set 0'
      pc=9, type=5, inst='i32.const 80'
      pc=12, type=5, inst='local.set 1'
      ... +87 more

**Implication**: Moderate graph (1294 blocks, 1981 edges). Per-edge identity embeddings (634-dim) suffice. No GNN needed. layout.json opcodes could enrich embeddings.

[Section 6: 0.1s]

>>> Running section 10

================================================================================
# Section 10: Sensor Quantization Feasibility
================================================================================

  Session: 2025-03-18_12-39-10 (572563 rows)

             Feature  #unique       Type    err@64   err@128   err@256   err@512  err@1024
-----------------------------------------------------------------------------------------------
      pendulum_state        3   discrete   0.7548%   0.1887%   0.0472%   0.0118%   0.0029%
           iteration       36   low-card   0.0253%   0.0065%   0.0016%   0.0004%   0.0001%
            target_x        4   discrete   0.0348%   0.0079%   0.0024%   0.0005%   0.0001%
           current_x   125057 continuous   0.0365%   0.0091%   0.0023%   0.0006%   0.0001%
            velocity   255587 continuous   0.4747%   0.1177%   0.0298%   0.0074%   0.0019%
       current_angle     5410 continuous   1.1002%   0.2080%   0.0365%   0.0106%   0.0025%
    angular_velocity      135 continuous 429.8405%  17.5331%  19.0970%   0.3304%   3.0831%

**Implication**: Max error at 256 bins = 19.10%. Consider 1024 bins or learned VQ-VAE for some features.

[Section 10: 0.6s]


Runtime: 0.0 minutes# Architecture-Informing EDA — CPS-Debugger FM Trace Generation
# 2026-04-09 00:55:59
# sections=3
================================================================================
Usable sessions: 29, Ctrl+math edges: 318, Variable: 32

>>> Running section 3

================================================================================
# Section 3: The 32 Variable Controller+Math Edges
================================================================================
  Scanning 2025-03-13_09-23-44 (34512 rows, sampling 1000)...
    collected 1000 ticks
  Scanning 2025-03-13_11-12-19 (186081 rows, sampling 1000)...
    collected 1000 ticks
  Scanning 2025-03-13_13-31-50 (69658 rows, sampling 1000)...
    collected 1000 ticks

  Combined matrix: (3000, 318)

  Variable: 105,  Constant: 213

                Edge               Function   Mean    Std  Min  Max  H(bits) Values
----------------------------------------------------------------------------------------------------
            20:58:60         stays_balanced   0.03   0.16    0    1    0.181 [0, 1]
          20:152:178         stays_balanced   0.03   0.16    0    1    0.181 [0, 1]
          20:179:201         stays_balanced   0.03   0.16    0    1    0.181 [0, 1]
        22:1092:1094                   tick   1.00   0.07    0    1    0.043 [0, 1]
        22:1092:1143                   tick   0.00   0.07    0    1    0.043 [0, 1]
        22:1141:1143                   tick   0.00   0.04    0    1    0.021 [0, 1]
        22:1141:1167                   tick   0.99   0.08    0    1    0.058 [0, 1]
        22:1209:1211                   tick   0.08   0.27    0    1    0.400 [0, 1]
        22:1209:1636                   tick   0.92   0.27    0    1    0.400 [0, 1]
        22:1287:1289                   tick   0.04   0.18    0    1    0.219 [0, 1]
        22:1287:1311                   tick   0.04   0.21    0    1    0.262 [0, 1]
        22:1309:1633                   tick   0.04   0.18    0    1    0.219 [0, 1]
        22:1375:1377                   tick   0.04   0.21    0    1    0.262 [0, 1]
        22:1397:1632                   tick   0.04   0.21    0    1    0.262 [0, 1]
        22:1634:2765                   tick   0.08   0.27    0    1    0.400 [0, 1]
        22:1676:1678                   tick   0.16   0.37    0    1    0.640 [0, 1]
        22:1676:1806                   tick   0.76   0.43    0    1    0.798 [0, 1]
        22:1714:1716                   tick   0.16   0.37    0    1    0.640 [0, 1]
        22:1889:1891                   tick   0.16   0.37    0    1    0.640 [0, 1]
        22:1889:1930                   tick   0.76   0.43    0    1    0.798 [0, 1]
        22:1928:2764                   tick   0.16   0.37    0    1    0.640 [0, 1]
        22:2093:2095                   tick   0.03   0.16    0    1    0.181 [0, 1]
        22:2093:2175                   tick   0.73   0.44    0    1    0.840 [0, 1]
        22:2173:2763                   tick   0.03   0.16    0    1    0.181 [0, 1]
        22:2283:2285                   tick   0.45   0.50    0    1    0.993 [0, 1]
        22:2283:2738                   tick   0.28   0.45    0    1    0.858 [0, 1]
        22:2368:2370                   tick   0.06   0.24    0    1    0.334 [0, 1]
        22:2368:2738                   tick   0.39   0.49    0    1    0.963 [0, 1]
        22:2488:2490                   tick   0.00   0.02    0    1    0.004 [0, 1]
        22:2488:2735                   tick   0.06   0.24    0    1    0.333 [0, 1]
        22:2681:2734                   tick   0.00   0.02    0    1    0.004 [0, 1]
        22:2736:2762                   tick   0.06   0.24    0    1    0.334 [0, 1]
            33:69:71               pou_main   0.84   0.37    0    1    0.640 [0, 1]
           33:69:131               pou_main   0.16   0.37    0    1    0.640 [0, 1]
          33:118:259               pou_main   0.84   0.37    0    1    0.640 [0, 1]
            34:19:21     turn_on_with_delay   1.76   0.66    0    3    1.428 [0, 1, 2, 3]
            34:19:46     turn_on_with_delay   1.24   0.66    0    3    1.428 [0, 1, 2, 3]
            34:24:26     turn_on_with_delay   0.35   0.48    0    1    0.933 [0, 1]
            34:24:46     turn_on_with_delay   1.41   0.52    0    2    1.081 [0, 1, 2]
            34:52:54     turn_on_with_delay   1.24   0.66    0    3    1.428 [0, 1, 2, 3]
           34:52:102     turn_on_with_delay   1.41   0.52    0    2    1.081 [0, 1, 2]
            34:57:59     turn_on_with_delay   0.90   0.63    0    3    1.364 [0, 1, 2, 3]
           34:57:102     turn_on_with_delay   0.34   0.47    0    1    0.926 [0, 1]
            34:71:73     turn_on_with_delay   0.41   0.54    0    2    1.094 [0, 1, 2]
            34:71:88     turn_on_with_delay   0.50   0.51    0    2    1.029 [0, 1, 2]
          34:112:114     turn_on_with_delay   0.34   0.47    0    1    0.926 [0, 1]
          34:112:129     turn_on_with_delay   1.41   0.52    0    2    1.081 [0, 1, 2]
          34:116:118     turn_on_with_delay   0.34   0.47    0    1    0.926 [0, 1]
            35:20:22    turn_off_with_delay   0.43   0.49    0    1    0.984 [0, 1]
            35:20:46    turn_off_with_delay   0.57   0.49    0    1    0.984 [0, 1]
            35:24:26    turn_off_with_delay   0.00   0.05    0    1    0.024 [0, 1]
            35:24:46    turn_off_with_delay   0.42   0.49    0    1    0.983 [0, 1]
            35:51:53    turn_off_with_delay   0.57   0.49    0    1    0.984 [0, 1]
           35:51:100    turn_off_with_delay   0.42   0.49    0    1    0.983 [0, 1]
            35:55:57    turn_off_with_delay   0.57   0.50    0    1    0.986 [0, 1]
           35:55:100    turn_off_with_delay   0.00   0.07    0    1    0.043 [0, 1]
            35:69:71    turn_off_with_delay   0.15   0.36    0    1    0.607 [0, 1]
            35:69:86    turn_off_with_delay   0.42   0.49    0    1    0.982 [0, 1]
          35:105:107    turn_off_with_delay   0.00   0.07    0    1    0.043 [0, 1]
          35:105:119    turn_off_with_delay   0.42   0.49    0    1    0.983 [0, 1]
          35:110:112    turn_off_with_delay   0.00   0.07    0    1    0.043 [0, 1]
            37:43:45    pou_general_machine   0.50   0.50    0    1    1.000 [0, 1]
            37:43:67    pou_general_machine   0.50   0.50    0    1    1.000 [0, 1]
            37:55:57    pou_general_machine   0.25   0.43    0    1    0.812 [0, 1]
            37:55:67    pou_general_machine   0.25   0.43    0    1    0.813 [0, 1]
          39:456:476        pou_standup_rel   0.16   0.37    0    1    0.640 [0, 1]
          39:495:497        pou_standup_rel   0.16   0.37    0    1    0.640 [0, 1]
          39:511:542        pou_standup_rel   0.16   0.37    0    1    0.640 [0, 1]
            40:44:46            pou_lqr_sim   0.01   0.09    0    1    0.074 [0, 1]
            40:44:85            pou_lqr_sim   0.75   0.43    0    1    0.812 [0, 1]
           40:83:185            pou_lqr_sim   0.01   0.09    0    1    0.074 [0, 1]
          40:281:283            pou_lqr_sim   0.01   0.09    0    1    0.074 [0, 1]
          40:281:316            pou_lqr_sim   0.75   0.43    0    1    0.812 [0, 1]
          40:299:301            pou_lqr_sim   0.01   0.09    0    1    0.074 [0, 1]
          40:314:340            pou_lqr_sim   0.01   0.09    0    1    0.074 [0, 1]
             63:8:10                   fmin   1.68   0.61    0    2    1.018 [0, 1, 2]
            63:17:22                   fmin   1.68   0.61    0    2    1.018 [0, 1, 2]
             64:8:10                   fmax   1.68   0.61    0    2    1.018 [0, 1, 2]
            64:17:22                   fmax   1.68   0.61    0    2    1.018 [0, 1, 2]
            72:53:55             __rem_pio2   0.14   0.35    0    1    0.590 [0, 1]
            72:67:69             __rem_pio2   0.14   0.35    0    1    0.590 [0, 1]
            72:80:82             __rem_pio2   0.09   0.29    0    1    0.441 [0, 1]
           72:80:206             __rem_pio2   0.05   0.22    0    1    0.289 [0, 1]
            72:89:91             __rem_pio2   0.05   0.21    0    1    0.272 [0, 1]
           72:89:148             __rem_pio2   0.04   0.21    0    1    0.263 [0, 1]
         72:146:1255             __rem_pio2   0.05   0.21    0    1    0.272 [0, 1]
         72:204:1255             __rem_pio2   0.04   0.21    0    1    0.263 [0, 1]
          72:214:216             __rem_pio2   0.03   0.16    0    1    0.170 [0, 1]
          72:214:273             __rem_pio2   0.03   0.16    0    1    0.170 [0, 1]
         72:271:1255             __rem_pio2   0.03   0.16    0    1    0.170 [0, 1]
         72:329:1255             __rem_pio2   0.03   0.16    0    1    0.170 [0, 1]
           73:86:110                  __sin   0.09   0.29    0    1    0.441 [0, 1]
            74:45:47                    cos   0.02   0.14    0    1    0.143 [0, 1]
            74:45:90                    cos   0.14   0.35    0    1    0.590 [0, 1]
            74:67:69                    cos   0.02   0.14    0    1    0.143 [0, 1]
           74:88:229                    cos   0.02   0.14    0    1    0.143 [0, 1]
          74:102:113                    cos   0.14   0.35    0    1    0.590 [0, 1]
          74:135:162                    cos   0.05   0.21    0    1    0.272 [0, 1]
          74:135:186                    cos   0.05   0.22    0    1    0.289 [0, 1]
          74:135:208                    cos   0.04   0.21    0    1    0.263 [0, 1]
          74:184:229                    cos   0.05   0.21    0    1    0.272 [0, 1]
          74:206:229                    cos   0.05   0.22    0    1    0.289 [0, 1]
            77:29:31                    log   0.07   0.29    0    2    0.386 [0, 1, 2]
           77:29:252                    log   0.25   0.62    0    2    0.769 [0, 1, 2]
          77:275:388                    log   0.25   0.62    0    2    0.769 [0, 1, 2]

  Entropy: mean=0.554, min=0.004, max=1.428 bits
  Total information per tick: 58.2 bits (7.3 bytes)

  Constant edges all fire exactly once? NO — deviations!

**Implication**: Prediction target is 32-dim with ~58 bits of info per tick. Very compact problem.

[Section 3: 10.8s]


Runtime: 0.2 minutes# Architecture-Informing EDA — CPS-Debugger FM Trace Generation
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


Runtime: 0.2 minutes# Architecture-Informing EDA — CPS-Debugger FM Trace Generation
# 2026-04-09 00:55:59
# sections=5,7
================================================================================
Usable sessions: 29, Ctrl+math edges: 318, Variable: 32

>>> Running section 5

================================================================================
# Section 5: Sensor-to-Trace Correlation (Feasibility Check)
================================================================================
  Merging 2025-03-13_09-23-44 (2000 ticks)...
    merged 2000 ticks
  Merging 2025-03-13_11-12-19 (2000 ticks)...
    merged 2000 ticks
  Merging 2025-03-13_13-31-50 (2000 ticks)...
    merged 2000 ticks

  Total merged: 6000
  Timestamp alignment: mean=233µs, max=207694µs

                Edge             Function        Best sensor     |r|
----------------------------------------------------------------------
          31:101:169      waxi_end_access   angular_velocity   0.998
            33:19:21             pou_main     pendulum_state   0.000
            33:19:46             pou_main     pendulum_state   0.000
            33:24:46             pou_main     pendulum_state   0.000
            33:52:54             pou_main     pendulum_state   0.000
           33:52:102             pou_main     pendulum_state   0.000
            33:57:59             pou_main     pendulum_state   0.000
           33:57:102             pou_main     pendulum_state   0.000
            33:71:73             pou_main     pendulum_state   0.000
          33:112:114             pou_main     pendulum_state   0.000
          33:112:129             pou_main     pendulum_state   0.000
          33:116:118             pou_main     pendulum_state   0.000
             34:8:10   turn_on_with_delay   angular_velocity   0.998
            34:19:21   turn_on_with_delay          current_x   0.328
            34:19:46   turn_on_with_delay          current_x   0.325
            34:24:46   turn_on_with_delay          current_x   0.407
            34:52:54   turn_on_with_delay          current_x   0.325
           34:52:102   turn_on_with_delay          current_x   0.407
            34:57:59   turn_on_with_delay          current_x   0.365
           34:57:102   turn_on_with_delay          iteration   0.076
            34:71:73   turn_on_with_delay          iteration   0.142
            34:71:88   turn_on_with_delay          current_x   0.398
          34:112:114   turn_on_with_delay          iteration   0.076
          34:112:129   turn_on_with_delay          current_x   0.407
          34:116:118   turn_on_with_delay          iteration   0.076
             63:8:10                 fmin          iteration   0.435
            63:17:22                 fmin          iteration   0.435
             64:8:10                 fmax          iteration   0.435
            64:17:22                 fmax          iteration   0.435
            77:29:31                  log     pendulum_state   0.488
           77:29:252                  log     pendulum_state   0.773
          77:275:388                  log     pendulum_state   0.773

Per-sensor summary:
        pendulum_state: max|r|=0.773, mean|r|=0.113
             iteration: max|r|=0.435, mean|r|=0.132
              target_x: max|r|=0.071, mean|r|=0.024
             current_x: max|r|=0.407, mean|r|=0.106
              velocity: max|r|=0.053, mean|r|=0.009
         current_angle: max|r|=0.074, mean|r|=0.022
      angular_velocity: max|r|=0.998, mean|r|=0.084

  Overall: max|r|=0.998, mean|r|=0.070

**Implication**: Strong linear signal exists. Sensor-conditioned prediction is feasible.

[Section 5: 11.8s]

>>> Running section 7

================================================================================
# Section 7: FSM State Distribution
================================================================================
  2025-03-13_09-23-44: {1: 17094, 0: 10556, 2: 6840}, 7 episodes
  2025-03-13_11-12-19: {1: 162316, 0: 14008, 2: 9735}, 11 episodes
  2025-03-13_13-31-50: {1: 60970, 0: 8666}, 2 episodes
  2025-03-13_14-32-27: {1: 102817, 0: 8325}, 2 episodes
  2025-03-17_10-06-14: {1: 481790, 0: 8261}, 2 episodes
  2025-03-17_10-21-19: {1: 475231, 0: 9373, 2: 2999}, 5 episodes
  2025-03-17_10-36-44: {1: 479525, 0: 8987, 2: 3387}, 5 episodes
  2025-03-17_10-51-58: {1: 467228, 0: 11488, 2: 6660}, 8 episodes
  2025-03-17_11-06-39: {1: 486294, 0: 8306}, 2 episodes
  2025-03-18_12-39-10: {1: 554001, 0: 11744, 2: 6818}, 8 episodes

Aggregate:
  pct_SWINGUP: mean=7.0%, std=9.1%
  pct_BALANCE: mean=90.2%, std=14.9%
  pct_RESET: mean=4.8%, std=7.5%

Per-state variable edge firing (from merged data):

  Ticks per state: SWINGUP=719, BALANCE=3413, RESET=368

                Edge             Function    SWINGUP    BALANCE      RESET
--------------------------------------------------------------------------------
          31:101:169      waxi_end_access      1.000      1.000      1.000
            33:19:21             pou_main      0.000      0.000      0.000
            33:19:46             pou_main      0.000      0.000      0.000
            33:24:46             pou_main      0.000      0.000      0.000
            33:52:54             pou_main      0.000      0.000      0.000
           33:52:102             pou_main      0.000      0.000      0.000
            33:57:59             pou_main      0.000      0.000      0.000
           33:57:102             pou_main      0.000      0.000      0.000
            33:71:73             pou_main      0.000      0.000      0.000
          33:112:114             pou_main      0.000      0.000      0.000
          33:112:129             pou_main      0.000      0.000      0.000
          33:116:118             pou_main      0.000      0.000      0.000
             34:8:10   turn_on_with_delay      3.000      3.000      3.000
            34:19:21   turn_on_with_delay      1.508      1.808      1.601
            34:19:46   turn_on_with_delay      1.492      1.192      1.399

**Implication**: FSM state should be an explicit conditioning input. If means differ substantially across states, consider state-specific model heads.

[Section 7: 12.5s]


Runtime: 0.4 minutes# Architecture-Informing EDA — CPS-Debugger FM Trace Generation
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


Runtime: 4.9 minutes# Architecture-Informing EDA — CPS-Debugger FM Trace Generation
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