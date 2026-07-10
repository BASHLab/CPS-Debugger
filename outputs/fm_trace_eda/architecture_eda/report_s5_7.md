# Architecture-Informing EDA — CPS-Debugger FM Trace Generation
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


Runtime: 0.4 minutes