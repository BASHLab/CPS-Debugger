# Execution Trace Data Inventory

**Purpose**: Inventory all trace data to support shifting from aggregate branch count prediction (regression) to sequence generation of ordered branch executions over time.

**Date**: 2026-04-08

---

## 1. Data Hierarchy (3 Granularity Levels)

### Level 1: Raw `.aspk` Binary Files (per-tick sequences)
- **Location**: `Pittsburgh-pendulum-datalogs/extracted/{session}/trace/*.aspk`
- **Format**: Binary, 8-byte entries, little-endian
  - **Bytes 0-7**: Timestamp header (uint64, microseconds since epoch)
  - **Bytes 8+**: Sequential branch transitions, each 8 bytes:
    - `[0:3]` target_pc (uint24, little-endian)
    - `[3:6]` source_pc (uint24, little-endian)
    - `[6:8]` source_function_index (uint16, little-endian)
  - Each entry = one `(func_id, from_pc, to_pc)` branch transition
- **Filename format**: `cc:{cycle_count}_ts:{timestamp_us}.aspk`
- **Cadence**: One file per ~1ms controller tick (1kHz control loop)
- **Sequence length per tick**: mean=1868, std=119, min=3, max=1898
- **Availability**: Only 7 of 13 extracted sessions (2025-03-17_10-51-58 through 2025-03-19_10-54-34). Earlier sessions only have processed_trace.csv.
- **Size**: ~7GB per session (~485K files of ~15KB each)
- **This is the key data for sequence generation**: ordered branch-by-branch execution within each 1ms tick.

### Level 2: `processed_trace.csv` (per-tick aggregates)
- **Location**: `Pittsburgh-pendulum-datalogs/extracted/{session}/processed_trace/processed_trace.csv`
- **Columns**: `timestamp`, `cf_table`, `function_graph_edges`
- **Cadence**: One row per ~1ms tick (same as raw .aspk)
- **Row count**: ~35K (short sessions) to ~505K (full sessions)
- **`cf_table`**: JSON nested dict `{func_id: {from_pc: {to_pc: count}}}` — **aggregate counts** of each edge per tick, NOT ordered sequences
  - Typical: ~27 functions active, ~215 unique edges, ~1870 total branch executions per tick
- **`function_graph_edges`**: JSON list of `[src_func_id, dst_func_id]` pairs — the **function call graph** for that tick (which functions called which)
- **Size**: 195MB (short) to 2.8GB (full) per session
- **Availability**: All 13 extracted sessions + feb04 session (14 total)

### Level 3: Aligned Parquet Dataset (10ms windows)
- **Location**: `CPS-Debugger/outputs/experiments/aligned_dataset.parquet`
- **Aggregation**: 10 processed_trace rows → 1 aligned window (sum of branch counts)
- **Shape**: 137,905 windows x 119 columns
- **Columns**:
  - Metadata: `win` (sequential window index), `run` (session ID)
  - Sensor (17): `dl_state`, `dl_target_x`, `dl_current_x_mean/std/delta`, `dl_angle_mean/std/delta`, `dl_velocity_mean`, `dl_ang_vel_mean/std`, `dl_iteration`, `sl_cpu/mem/load1/load5/load15`
  - Trace edges (100): format `func_id:from_pc:to_pc`, values are summed counts over 10ms
- **Sessions included**: 6 of 13 (2025-03-13_09-23-44 through 2025-03-17_10-21-19)
- **Edge selection**: 100 of 306 unique raw edges (filtered by variance/frequency)

---

## 2. Session Manifest

### Extracted Sessions (13 in `extracted/`)

| Session | Duration(s) | PT Rows | Raw .aspk | Datalayer | Layout | In Aligned? |
|---------|-------------|---------|-----------|-----------|--------|-------------|
| 2025-03-13_09-23-44 | 34.7 | 34,512 | No | Yes | Yes | Yes (3,450 win) |
| 2025-03-13_11-12-19 | 186.3 | 186,081 | No | Yes | Yes | Yes (18,608 win) |
| 2025-03-13_13-31-50 | 69.8 | 69,658 | No | Yes | Yes | Yes (6,965 win) |
| 2025-03-13_14-32-27 | 111.4 | 111,164 | No | Yes | Yes | Yes (11,115 win) |
| 2025-03-17_10-06-14 | 490.3 | 490,073 | No | Yes | Yes | Yes (49,006 win) |
| 2025-03-17_10-21-19 | 487.8 | 487,625 | No | Yes | Yes | Yes (48,761 win) |
| 2025-03-17_10-51-58 | 485.6 | 485,398 | **Yes (485K)** | Yes | Yes | No |
| 2025-03-17_11-06-39 | 494.8 | 494,622 | **Yes (495K)** | Yes | Yes | No |
| 2025-03-18_12-39-10 | 505.4 | 505,175 | **Yes (505K)** | Yes | Yes | No |
| 2025-03-19_10-05-47 | 487.8 | 487,608 | **Yes (488K)** | Yes | Yes | No |
| 2025-03-19_10-20-35 | 488.6 | 488,453 | **Yes (488K)** | Yes | Yes | No |
| 2025-03-19_10-37-56 | 488.9 | 488,725 | **Yes (489K)** | Yes | Yes | No |
| 2025-03-19_10-54-34 | — | MISSING | Yes (485K) | **No** | Yes | No |

### Additional Sessions
- **2025-02-04_09-33-16** (in `extracted_feb04/`): 36,050 rows, 36.2s, processed_trace only
- **17 more sessions** listed in `eda/pendulum_webdataset.ipynb` `good_sessions` (2025-03-19 to 2025-03-26) — NOT in `extracted/`, likely in `2025.zip` (17GB archive)
- **2025-01-13_10-58-23.tar.gz**: untested archive at top level

**Total known sessions**: 14 extracted + 17 in archive = ~31 unique sessions

---

## 3. Edge Vocabulary

### Raw Vocabulary (from 500 sampled ticks, session 2025-03-17_10-51-58)
- **306 unique edge triplets** `(func_id, from_pc, to_pc)`
- Dominated by library functions: `printf_core` (func 102) and `memcpy` (func 108) account for >50% of branch executions
- Top edge: `(102, 223, 225)` with ~195 executions per tick

### Aligned Parquet Vocabulary
- **100 edges** selected from the 306 (filtered by variance/frequency during alignment)
- 15 functions represented: 22(tick), 33(pou_main), 34(turn_on_with_delay), 35(turn_off_with_delay), 37(pou_general_machine), 40(pou_lqr_sim), 63(fmin), 64(fmax), 83(writev), 84(__stdio_write), 92(__fwritex), 101(vfprintf), 102(printf_core), 108(memcpy), 109(memset)
- **44 controller edges, 56 library edges**

### Functions per Category
- **Controller logic** (11 funcs): tick(22), pou_main(33), turn_on/off(34,35), pou_general_machine(37), pou_general_drive(38), pou_standup_rel(39), pou_lqr_sim(40), pou_drive_control_word(42), fmin(63), fmax(64)
- **Library/IO** (remaining): printf_core(102), memcpy(108), memset(109), __fwritex(92), etc.
- **Key insight**: For FM sequence generation, may want to filter to controller-only edges to reduce vocab size and focus on behaviorally-relevant patterns

---

## 4. layout.json Structure

- **Location**: `{session}/code/layout.json`
- **Format**: JSON dict with keys:
  - Numeric strings (func IDs): each maps to `{name, br_blocks, call_direct_sites, call_indirect_sites, first_block, target_sites}`
  - `func_exports`: `{"_start": 19, "execute": 21}`
  - `import_func_count`: number of imported (host) functions
- **113 functions** defined, ~60 with non-trivial branch blocks
- **`br_blocks`**: Maps `from_pc` → list of instructions in that basic block. Each instruction has `{inst, pc, type}`. The `type=0` entries are the actual branch instructions (`br_if`, `br`).
- **`call_direct_sites`**: Maps call-site PC → target function ID (static call graph)
- **Same layout.json across all sessions** (same compiled Wasm module)

---

## 5. Sequence Generation Requirements

### What we're generating
Instead of predicting a 100-dim count vector per 10ms window, generate the **ordered sequence of branch executions** within each 1ms tick.

### Key parameters
| Parameter | Value |
|-----------|-------|
| Sequence length per tick | ~1870 tokens (mean), up to 1898 |
| Vocabulary size (all edges) | 306 unique edge triplets |
| Vocabulary size (controller only) | ~100 edges |
| Ticks per session | 35K–505K |
| Sessions with raw sequences | 7 (extracted) + potentially 17 more (in 2025.zip) |
| Sessions with aggregate counts only | 7 (no raw .aspk) |
| Total raw sequences available | ~3.4M ticks (7 sessions × ~490K each) |

### Encoding options for tokens
1. **Flat triplet**: `(func_id, from_pc, to_pc)` as a single token → vocab=306
2. **Hierarchical**: separate func_id + edge_within_func tokens → smaller per-position vocab
3. **Controller-filtered**: drop library edges (printf, memcpy) → vocab≈100, seq_len≈300-500

### Data availability gap
- Raw .aspk files (needed for sequences) exist for only 7/14 extracted sessions
- 17 additional sessions in `2025.zip` — unknown if they contain raw .aspk
- processed_trace.csv only provides **counts per edge**, not ordering — **cannot reconstruct sequences** from Level 2 data
- **Recommendation**: Extract `2025.zip` to check for raw .aspk in the 17 additional sessions

---

## 6. Cross-Session Consistency

- **All sessions share the same Wasm module** (identical layout.json, same compiled code)
- **cf_table structure is consistent**: ~27 active functions, ~215 edges per tick
- **Timestamp format**: microseconds since epoch, consistent across sessions
- **Anomaly**: Session 2025-03-19_10-54-34 has .aspk files but no datalayer.csv and no processed_trace.csv — incomplete extraction
- **Edge set may vary by controller state**: SWINGUP vs BALANCE activates different code paths → different edges per tick. The 306-edge vocabulary is the union across all states.

---

## 7. Prior Work (from EDA Notebooks)

### Pendulum EDA.ipynb
- Loaded one session's processed_trace.csv, merged with datalayer on timestamp
- Defined `parse_edges()` to extract function_graph_edges as `(src, dst)` tuples
- Built `extract_trace_features()` for windowed edge count vectors
- UMAP/PCA visualization of combined datalayer + trace features
- Multi-session vocabulary union across 30 sessions

### pendulum_webdataset.ipynb
- 30 "good_sessions" identified
- Built triplet_to_idx mapping: `(func_id, from_pc, to_pc)` → integer index
- Created multimodal dataset: video frames + datalayer + trace count vectors + pendulum state labels
- Window size: 15 frames at 2 windows/sec

---

## 8. Recommendations for FM Trace Generation

1. **Extract 2025.zip** to check for raw .aspk in 17 additional sessions — could triple the training data
2. **Start with 7 sessions that have raw .aspk** for sequence generation prototyping
3. **Consider controller-only filtering**: dropping library functions (102, 108, 109, 92, etc.) reduces vocab from 306→~100 and seq_len from ~1870→~300-500, making generation much more tractable
4. **Condition on sensor state**: the datalayer provides `dl_state` (0=SWINGUP, 1=BALANCE, 2=RESET) and continuous sensor readings that should condition the trace generation
5. **Temporal structure**: sequences within a tick show the full control loop execution; sequences across ticks show how the execution pattern evolves as the physical system changes state
6. **Validation approach**: compare generated sequences against ground truth using (a) edge count accuracy (does the count vector match?), (b) sequence ordering metrics, (c) subsequence coverage
