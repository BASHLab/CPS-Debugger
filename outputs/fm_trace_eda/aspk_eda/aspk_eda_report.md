# ASPK Binary Trace Format EDA
# Session: 2025-03-18_12-39-10
# Generated: 2026-04-09 10:58:38
# Format: 8-byte header + N x 8B entries (to_pc:u24, from_pc:u24, func_id:u16) LE

Listing aspk files...
  505175 aspk files found

================================================================================
# Section 1: Binary Format Verification
================================================================================

Format hypothesis: 8-byte header + N x 8-byte entries
  Each entry: bytes[0:3]=to_pc (u24 LE), bytes[3:6]=from_pc (u24 LE), bytes[6:8]=func_id (u16 LE)

10 sample files (size, decoded entries):
    size_B         cc                   ts  n_entries  divisible_by_8
  ----------------------------------------------------------------------
        32    2813977     1742315886842718          3  True
     15024    2909927     1742315982988975       1877  True
     15040    2917502     1742315990563978       1879  True
     15056    3031127     1742316104188979       1881  True
     15168    2816502     1742315889563977       1895  True
     15008    2942752     1742316015813978       1875  True
     15032    2836702     1742315909763972       1878  True
     15040    3311402     1742316384463989       1879  True
     15064    2978102     1742316051163978       1882  True
     15160    3278577     1742316351638988       1894  True

--- Cross-reference verification against processed_trace ---
  ts=1742316165704980  PT_total=1876  aspk_total=1876  unique=213  exact_match_unique=213/213  perfect_dict_eq=True
  ts=1742316020857979  PT_total=1895  aspk_total=1895  unique=217  exact_match_unique=217/217  perfect_dict_eq=True
  ts=1742315893112974  PT_total=1883  aspk_total=1883  unique=224  exact_match_unique=224/224  perfect_dict_eq=True
  ts=1742316066051979  PT_total=1878  aspk_total=1878  unique=218  exact_match_unique=218/218  perfect_dict_eq=True
  ts=1742316224557984  PT_total=1880  aspk_total=1880  unique=217  exact_match_unique=217/217  perfect_dict_eq=True

  Verified 20 aspk files against processed_trace, 20 perfect matches.

--- 3 fully decoded example ticks ---

  cc=2864494 ts=1742315937555975 size=15016B  n_entries=1876
  First 12 entries (func_id, func_name, from_pc -> to_pc):
    [  0]  func= 21 execute                       101 -> 113   
    [  1]  func= 21 execute                       155 -> 201   
    [  2]  func= 21 execute                       251 -> 297   
    [  3]  func= 21 execute                       348 -> 350   
    [  4]  func= 47 __clock_gettime                43 -> 62    
    [  5]  func= 22 tick                          209 -> 253   
    [  6]  func= 26 ethercat_read                 156 -> 247   
    [  7]  func= 26 ethercat_read                 372 -> 490   
    [  8]  func= 26 ethercat_read                 636 -> 755   
    [  9]  func= 26 ethercat_read                1022 -> 1153  
    [ 10]  func= 27 waxi_end_access               101 -> 169   
    [ 11]  func=109 memset                          8 -> 10    
    ... (1864 more)

  cc=3066564 ts=1742316139625981 size=15056B  n_entries=1881
  First 12 entries (func_id, func_name, from_pc -> to_pc):
    [  0]  func= 21 execute                       101 -> 113   
    [  1]  func= 21 execute                       155 -> 201   
    [  2]  func= 21 execute                       251 -> 297   
    [  3]  func= 21 execute                       348 -> 350   
    [  4]  func= 47 __clock_gettime                43 -> 62    
    [  5]  func= 22 tick                          209 -> 253   
    [  6]  func= 26 ethercat_read                 156 -> 247   
    [  7]  func= 26 ethercat_read                 372 -> 490   
    [  8]  func= 26 ethercat_read                 636 -> 755   
    [  9]  func= 26 ethercat_read                1022 -> 1153  
    [ 10]  func= 27 waxi_end_access               101 -> 169   
    [ 11]  func=109 memset                          8 -> 10    
    ... (1869 more)

  cc=3268634 ts=1742316341695984 size=15048B  n_entries=1880
  First 12 entries (func_id, func_name, from_pc -> to_pc):
    [  0]  func= 21 execute                       101 -> 113   
    [  1]  func= 21 execute                       155 -> 201   
    [  2]  func= 21 execute                       251 -> 297   
    [  3]  func= 21 execute                       348 -> 350   
    [  4]  func= 47 __clock_gettime                43 -> 62    
    [  5]  func= 22 tick                          209 -> 253   
    [  6]  func= 26 ethercat_read                 156 -> 247   
    [  7]  func= 26 ethercat_read                 372 -> 490   
    [  8]  func= 26 ethercat_read                 636 -> 755   
    [  9]  func= 26 ethercat_read                1022 -> 1153  
    [ 10]  func= 27 waxi_end_access               101 -> 169   
    [ 11]  func=109 memset                          8 -> 10    
    ... (1868 more)

================================================================================
# Section 2: Sequence Length Distribution (10,000 uniform samples)
================================================================================

Saved 10000 samples to seq_length_dist.csv

  ALL entries per tick:
    n=10000  min=3  max=1903  mean=1875.6  median=1879  std=23.8
    p5=1868  p25=1876  p75=1881  p95=1884  p99=1893

  Controller+Math entries per tick:
    n=10000  min=3  max=101  mean=91.0  median=91  std=2.6
    p5=88  p25=90  p75=93  p95=94  p99=98

  Library entries per tick:
    n=10000  min=0  max=1809  mean=1784.7  median=1788  std=23.0
    p5=1783  p25=1786  p75=1789  p95=1792  p99=1802

  Saved histogram → fig_seq_length_aspk.png

================================================================================
# Section 3: Edge Triple Vocabulary
================================================================================

  Unique edge triples: 318
    Controller+Math: 167  (raw count: 909633)
    Library:         151  (raw count: 17846647)
  Token vocabulary size if generating triples directly: 318

  Saved vocabulary → edge_vocabulary.csv

  Top 20 most frequent edge triples:
    rank func func_name                   from->to           count  cat
    ----------------------------------------------------------------------
       1  102 printf_core                  223->225        1959804  lib
       2  102 printf_core                  230->822        1719828  lib
       3  102 printf_core                  837->206        1719828  lib
       4  108 memcpy                        12->14          407970  lib
       5   92 __fwritex                     12->35          398407  lib
       6   92 __fwritex                     51->72          397817  lib
       7   92 __fwritex                     87->192         397817  lib
       8  108 memcpy                       225->504         315095  lib
       9  108 memcpy                       520->679         315095  lib
      10  108 memcpy                        20->22          298596  lib
      11  108 memcpy                       772->774         290722  lib
      12  108 memcpy                       774->903         290722  lib
      13  108 memcpy                       947->973         290722  lib
      14  102 printf_core                  165->167         289971  lib
      15  108 memcpy                       689->691         277253  lib
      16  108 memcpy                       699->766         277253  lib
      17  108 memcpy                       980->992         273165  lib
      18  108 memcpy                       912->938         264352  lib
      19  102 printf_core                  200->202         259974  lib
      20  102 printf_core                  319->321         259974  lib

================================================================================
# Section 4: Timestamp Alignment to Datalayer
================================================================================

  Datalayer: 572563 rows
    ts_us range: 1742315887050847 … 1742316459612860

  Aligned 100 aspk files:
    mean alignment error = 2207.7 µs
    max  alignment error = 208129 µs
    std  alignment error = 20695.9 µs
    p50/p95/p99 = 128 / 130 / 2210 µs

  3 alignment examples (with sensor values):

    aspk_ts=1742315886842718  dl_ts=1742315887050847  delta=-208129µs  cc=2813977
      pendulum_state     = 0.0
      iteration          = 0.0
      target_x           = 0.0
      current_x          = 0.001406
      velocity           = 0.0
      current_angle      = -3.141593
      angular_velocity   = -3141.592653

    aspk_ts=1742315892140976  dl_ts=1742315892140845  delta=+131µs  cc=2819079
      pendulum_state     = 0.0
      iteration          = 0.0
      target_x           = 0.0
      current_x          = 0.062509
      velocity           = -0.163241
      current_angle      = -2.445932
      angular_velocity   = 0.0

    aspk_ts=1742315897243975  dl_ts=1742315897243846  delta=+129µs  cc=2824182
      pendulum_state     = 0.0
      iteration          = 0.0
      target_x           = 0.0
      current_x          = -0.003168
      velocity           = 0.073572
      current_angle      = 0.469398
      angular_velocity   = -9.203885

================================================================================
# Section 5: Ordering Structure & CFG Validity
================================================================================

  Loaded 634 known valid edges from inventory

--- Full ordered controller+math sequences (5 example ticks) ---

  cc=2898172  total=1877  ctrl_math=92
  Function-id sequence (first 40):
    21 21 21 21 22 26 26 26 26 27 26 22 22 35 35 35 35 35 34 34 34 34 34 37 22 22 38 34 34 34 34 34 34 34 34 34 34 36 36 33
  Full edge sequence (first 15):
    [  0]  func= 21 execute                       101 -> 113   
    [  1]  func= 21 execute                       155 -> 201   
    [  2]  func= 21 execute                       251 -> 297   
    [  3]  func= 21 execute                       348 -> 350   
    [  4]  func= 22 tick                          209 -> 253   
    [  5]  func= 26 ethercat_read                 156 -> 247   
    [  6]  func= 26 ethercat_read                 372 -> 490   
    [  7]  func= 26 ethercat_read                 636 -> 755   
    [  8]  func= 26 ethercat_read                1022 -> 1153  
    [  9]  func= 27 waxi_end_access               101 -> 169   
    [ 10]  func= 26 ethercat_read                1381 -> 1383  
    [ 11]  func= 22 tick                          336 -> 418   
    [ 12]  func= 22 tick                          438 -> 602   
    [ 13]  func= 35 turn_off_with_delay             8 -> 10    
    [ 14]  func= 35 turn_off_with_delay            20 -> 46    

  cc=2982368  total=1881  ctrl_math=93
  Function-id sequence (first 40):
    21 21 21 21 22 26 26 26 26 27 26 22 22 35 35 35 35 35 34 34 34 34 34 37 37 22 22 38 34 34 34 34 34 34 34 34 34 34 36 36
  Full edge sequence (first 15):
    [  0]  func= 21 execute                       101 -> 113   
    [  1]  func= 21 execute                       155 -> 201   
    [  2]  func= 21 execute                       251 -> 297   
    [  3]  func= 21 execute                       348 -> 350   
    [  4]  func= 22 tick                          209 -> 253   
    [  5]  func= 26 ethercat_read                 156 -> 247   
    [  6]  func= 26 ethercat_read                 372 -> 490   
    [  7]  func= 26 ethercat_read                 636 -> 755   
    [  8]  func= 26 ethercat_read                1022 -> 1153  
    [  9]  func= 27 waxi_end_access               101 -> 169   
    [ 10]  func= 26 ethercat_read                1381 -> 1383  
    [ 11]  func= 22 tick                          336 -> 418   
    [ 12]  func= 22 tick                          438 -> 602   
    [ 13]  func= 35 turn_off_with_delay             8 -> 10    
    [ 14]  func= 35 turn_off_with_delay            20 -> 46    

  cc=3066564  total=1881  ctrl_math=94
  Function-id sequence (first 40):
    21 21 21 21 22 26 26 26 26 27 26 22 22 35 35 35 35 35 34 34 34 34 34 37 37 22 22 38 34 34 34 34 34 34 34 34 34 34 34 36
  Full edge sequence (first 15):
    [  0]  func= 21 execute                       101 -> 113   
    [  1]  func= 21 execute                       155 -> 201   
    [  2]  func= 21 execute                       251 -> 297   
    [  3]  func= 21 execute                       348 -> 350   
    [  4]  func= 22 tick                          209 -> 253   
    [  5]  func= 26 ethercat_read                 156 -> 247   
    [  6]  func= 26 ethercat_read                 372 -> 490   
    [  7]  func= 26 ethercat_read                 636 -> 755   
    [  8]  func= 26 ethercat_read                1022 -> 1153  
    [  9]  func= 27 waxi_end_access               101 -> 169   
    [ 10]  func= 26 ethercat_read                1381 -> 1383  
    [ 11]  func= 22 tick                          336 -> 418   
    [ 12]  func= 22 tick                          438 -> 602   
    [ 13]  func= 35 turn_off_with_delay             8 -> 10    
    [ 14]  func= 35 turn_off_with_delay            20 -> 22    

  cc=3150760  total=1876  ctrl_math=90
  Function-id sequence (first 40):
    21 21 21 21 22 26 26 26 26 27 26 22 22 35 35 35 35 35 34 34 34 34 34 37 22 22 38 34 34 34 34 34 34 34 34 34 34 36 36 33
  Full edge sequence (first 15):
    [  0]  func= 21 execute                       101 -> 113   
    [  1]  func= 21 execute                       155 -> 201   
    [  2]  func= 21 execute                       251 -> 297   
    [  3]  func= 21 execute                       348 -> 350   
    [  4]  func= 22 tick                          209 -> 253   
    [  5]  func= 26 ethercat_read                 156 -> 247   
    [  6]  func= 26 ethercat_read                 372 -> 490   
    [  7]  func= 26 ethercat_read                 636 -> 755   
    [  8]  func= 26 ethercat_read                1022 -> 1153  
    [  9]  func= 27 waxi_end_access               101 -> 169   
    [ 10]  func= 26 ethercat_read                1381 -> 1383  
    [ 11]  func= 22 tick                          336 -> 418   
    [ 12]  func= 22 tick                          438 -> 602   
    [ 13]  func= 35 turn_off_with_delay             8 -> 10    
    [ 14]  func= 35 turn_off_with_delay            20 -> 46    

  cc=3234956  total=1879  ctrl_math=93
  Function-id sequence (first 40):
    21 21 21 21 22 26 26 26 26 27 26 22 22 35 35 35 35 35 34 34 34 34 34 37 37 22 22 38 34 34 34 34 34 34 34 34 34 34 36 36
  Full edge sequence (first 15):
    [  0]  func= 21 execute                       101 -> 113   
    [  1]  func= 21 execute                       155 -> 201   
    [  2]  func= 21 execute                       251 -> 297   
    [  3]  func= 21 execute                       348 -> 350   
    [  4]  func= 22 tick                          209 -> 253   
    [  5]  func= 26 ethercat_read                 156 -> 247   
    [  6]  func= 26 ethercat_read                 372 -> 490   
    [  7]  func= 26 ethercat_read                 636 -> 755   
    [  8]  func= 26 ethercat_read                1022 -> 1153  
    [  9]  func= 27 waxi_end_access               101 -> 169   
    [ 10]  func= 26 ethercat_read                1381 -> 1383  
    [ 11]  func= 22 tick                          336 -> 418   
    [ 12]  func= 22 tick                          438 -> 602   
    [ 13]  func= 35 turn_off_with_delay             8 -> 10    
    [ 14]  func= 35 turn_off_with_delay            20 -> 46    

--- Function-order consistency across ticks ---
  All 5 ticks share the same function-id at positions 0..23  (then diverge)
  Identical multisets of ctrl+math funcs across 5 ticks: False

--- Position-wise function entropy (across 1000 sampled ticks) ---
  Sampled 1000 ticks; ctrl+math seq length: median=91, max=100
  Position 0..199: 26/200 positions have a CONSTANT func across all ticks
  Position-wise entropy (first 30 positions, |unique|/N):
    p0=1  p1=1  p2=1  p3=1  p4=1  p5=1  p6=1  p7=1  p8=1  p9=1  p10=1  p11=1  p12=1  p13=1  p14=1  p15=1  p16=2  p17=2  p18=2  p19=1  p20=1  p21=2  p22=3  p23=3  p24=3  p25=3  p26=3  p27=3  p28=2  p29=1

--- CFG-validity of consecutive edge pairs ---
  Testing: for consecutive entries e1=(f1,fp1,tp1), e2=(f2,fp2,tp2),
    intra-function pairs (f1==f2): is fp2 == tp1? (sequential CFG step)
    inter-function pairs (f1!=f2): always allowed (call/return)
  Pairs examined: 373075
  Edges that are in inventory: 373073/373075  (100.00%)
  Intra-function pairs: 333357 (89.35%)
  Intra-function pairs that are sequential CFG steps (fp2==tp1): 5787/333357 (1.74%)


Total runtime: 1.1 min
