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


Runtime: 0.0 minutes