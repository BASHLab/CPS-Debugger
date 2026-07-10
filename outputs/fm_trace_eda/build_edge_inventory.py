#!/usr/bin/env python3
"""Build definitive edge inventory across all available sessions.

Reads cf_table from processed_trace.csv (sampled) for every session.
Produces:
  - full_session_manifest.csv
  - all_edges_inventory.csv
  - controller_edges.txt / .json
  - library_edges.txt / .json
  - func_id_to_name.json
"""

import json
import hashlib
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

BASE = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs")
OUT_DIR = Path("/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_trace_eda")

SAMPLE_TICKS_PER_SESSION = 1000

# --- Controller vs library classification ---
CONTROLLER_FUNCS = {
    'tick', 'execute', 'stays_balanced',
    'pou_main', 'pou_general_machine', 'pou_general_drive', 'pou_general_axis',
    'pou_standup_rel', 'pou_lqr_sim', 'pou_drive_control_word',
    'turn_on_with_delay', 'turn_off_with_delay',
    'log_data', 'set_end_of_process_flag',
    'ethercat_read', 'ethercat_write',
    'waxi_end_access',
}
MATH_FUNCS = {'fmin', 'fmax', 'cos', 'sin', 'log', 'sqrt', 'scalbn', 'frexp',
              '__rem_pio2', '__rem_pio2_large', '__sin'}


def classify_function(func_name):
    if func_name in CONTROLLER_FUNCS:
        return 'controller'
    elif func_name in MATH_FUNCS:
        return 'math'
    else:
        return 'library'


def inventory_session(sess_dir):
    """Inventory one session directory."""
    sess_dir = Path(sess_dir)
    entry = {'session_id': sess_dir.name, 'path': str(sess_dir)}

    pt = sess_dir / "processed_trace" / "processed_trace.csv"
    if pt.exists():
        entry['has_processed_trace'] = True
        entry['processed_trace_path'] = str(pt)
        entry['processed_trace_size_mb'] = round(pt.stat().st_size / 1e6, 1)
        with open(pt, 'rb') as f:
            entry['processed_trace_rows'] = sum(1 for _ in f) - 1
    else:
        entry['has_processed_trace'] = False
        entry['processed_trace_rows'] = 0

    trace_dir = sess_dir / "trace"
    entry['has_raw_aspk'] = trace_dir.is_dir() and any(trace_dir.glob("*.aspk"))

    layout = sess_dir / "code" / "layout.json"
    entry['has_layout'] = layout.exists()
    if layout.exists():
        entry['layout_path'] = str(layout)
        with open(layout, 'rb') as f:
            entry['layout_md5'] = hashlib.md5(f.read()).hexdigest()

    # Hash module.wasm if present (actual compiled binary)
    wasm = sess_dir / "code" / "module.wasm"
    entry['has_wasm'] = wasm.exists()
    if wasm.exists():
        entry['wasm_path'] = str(wasm)
        entry['wasm_size'] = wasm.stat().st_size
        with open(wasm, 'rb') as f:
            entry['wasm_md5'] = hashlib.md5(f.read()).hexdigest()

    # Hash module.wat if present (text format)
    wat = sess_dir / "code" / "module.wat"
    entry['has_wat'] = wat.exists()

    dl = sess_dir / "datalayer" / "datalayer.csv"
    entry['has_datalayer'] = dl.exists()

    return entry


def load_layout(layout_path):
    with open(layout_path) as f:
        layout = json.load(f)
    func_map = {}
    for k, v in layout.items():
        if k.isdigit() and isinstance(v, dict) and 'name' in v:
            func_map[int(k)] = v['name']
    return func_map


def main():
    # ---- Step 1: Session manifest ----
    print("=== Step 1: Building session manifest ===")
    all_sessions = []

    for d in sorted(BASE.glob("extracted/2025-*")):
        if d.is_dir():
            all_sessions.append(inventory_session(d))

    feb_dir = BASE / "extracted_feb04" / "data" / "2025-02-04_09-33-16"
    if feb_dir.is_dir():
        all_sessions.append(inventory_session(feb_dir))

    manifest = pd.DataFrame(all_sessions)
    manifest.to_csv(OUT_DIR / "full_session_manifest.csv", index=False)
    print(f"  {len(manifest)} sessions inventoried")
    print(f"  With processed_trace: {manifest['has_processed_trace'].sum()}")
    print(f"  With raw .aspk: {manifest['has_raw_aspk'].sum()}")
    print(f"  With datalayer: {manifest['has_datalayer'].sum()}")

    # ---- Step 2: Wasm build / layout consistency ----
    print("\n=== Step 2: Wasm build versions ===")

    # Group by layout.json hash
    layout_hashes = manifest[manifest['has_layout']]['layout_md5'].unique()
    print(f"  Unique layout.json versions: {len(layout_hashes)}")
    for h in layout_hashes:
        sessions = manifest[manifest['layout_md5'] == h]['session_id'].tolist()
        n_pt = manifest[(manifest['layout_md5'] == h) & (manifest['has_processed_trace'])].shape[0]
        print(f"    layout {h[:12]}: {len(sessions)} sessions ({n_pt} with processed_trace)")
        for s in sessions:
            print(f"      {s}")

    # Group by module.wasm hash (the actual binary)
    if 'wasm_md5' in manifest.columns:
        wasm_hashes = manifest[manifest['has_wasm']]['wasm_md5'].unique()
        print(f"\n  Unique module.wasm binaries: {len(wasm_hashes)}")
        for h in wasm_hashes:
            sub = manifest[manifest['wasm_md5'] == h]
            sz = sub['wasm_size'].iloc[0]
            sessions = sub['session_id'].tolist()
            n_pt = sub[sub['has_processed_trace']].shape[0]
            print(f"    wasm {h[:12]} ({sz/1e6:.1f} MB): {len(sessions)} sessions ({n_pt} with pt)")
            for s in sessions:
                print(f"      {s}")
    else:
        print("  No module.wasm files found.")

    # Load function maps from ALL unique layouts (they may differ)
    all_func_maps = {}
    for h in layout_hashes:
        row = manifest[manifest['layout_md5'] == h].iloc[0]
        fm = load_layout(row['layout_path'])
        all_func_maps[h] = fm
        print(f"\n  Layout {h[:12]}: {len(fm)} functions")

    # Use the most common layout as canonical
    most_common_hash = manifest[manifest['has_layout']]['layout_md5'].value_counts().idxmax()
    func_map = all_func_maps[most_common_hash]
    print(f"\n  Canonical layout (most sessions): {most_common_hash[:12]} ({len(func_map)} functions)")

    # Check for function name differences across layouts
    all_names = set()
    for fm in all_func_maps.values():
        all_names.update(fm.values())
    print(f"  Union of all function names: {len(all_names)}")

    # Save func map
    with open(OUT_DIR / "func_id_to_name.json", "w") as f:
        json.dump({str(k): v for k, v in sorted(func_map.items())}, f, indent=2)

    # ---- Step 3: Edge vocabulary from all sessions ----
    print("\n=== Step 3: Building edge vocabulary ===")
    edge_counter = Counter()
    edge_sessions = defaultdict(set)
    edge_per_tick = defaultdict(list)
    sessions_processed = 0

    pt_sessions = manifest[manifest['has_processed_trace']]
    for _, row in pt_sessions.iterrows():
        sid = row['session_id']
        pt_path = row['processed_trace_path']
        total_rows = row['processed_trace_rows']
        print(f"  {sid} ({total_rows} rows)...", end=" ", flush=True)

        try:
            total_rows_int = int(total_rows)
            N = SAMPLE_TICKS_PER_SESSION
            if total_rows_int > N * 3:
                # Read 3 chunks: head, middle, tail
                head = pd.read_csv(pt_path, nrows=N)
                mid_start = total_rows_int // 2 - N // 2
                mid = pd.read_csv(pt_path, skiprows=range(1, mid_start + 1), nrows=N)
                tail_start = total_rows_int - N
                tail = pd.read_csv(pt_path, skiprows=range(1, tail_start + 1), nrows=N)
                sample = pd.concat([head, mid, tail], ignore_index=True)
            else:
                sample = pd.read_csv(pt_path)

            session_edges = set()
            for _, tick_row in sample.iterrows():
                cf = json.loads(tick_row['cf_table'])
                for fid_str, pcs in cf.items():
                    fid = int(fid_str)
                    for from_str, targets in pcs.items():
                        fp = int(from_str)
                        for to_str, count in targets.items():
                            tp = int(to_str)
                            edge = (fid, fp, tp)
                            edge_counter[edge] += count
                            edge_sessions[edge].add(sid)
                            edge_per_tick[edge].append(count)
                            session_edges.add(edge)

            sessions_processed += 1
            print(f"{len(session_edges)} edges, {len(sample)} ticks sampled")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n  Processed {sessions_processed} sessions")
    print(f"  Total unique edges: {len(edge_counter)}")

    # ---- Step 4: Classify and build DataFrame ----
    print("\n=== Step 4: Classifying edges ===")
    records = []
    for edge, total_count in edge_counter.most_common():
        fid, fp, tp = edge
        fname = func_map.get(fid, f"unknown_{fid}")
        cat = classify_function(fname)
        ptick = edge_per_tick[edge]
        ptick_arr = np.array(ptick, dtype=float)
        mean_pt = ptick_arr.mean()

        records.append({
            'edge_id': f'{fid}:{fp}:{tp}',
            'func_id': fid,
            'func_name': fname,
            'from_pc': fp,
            'to_pc': tp,
            'category': cat,
            'total_count': total_count,
            'n_sessions': len(edge_sessions[edge]),
            'mean_per_tick': round(mean_pt, 4),
            'std_per_tick': round(ptick_arr.std(), 4),
            'min_per_tick': int(ptick_arr.min()),
            'max_per_tick': int(ptick_arr.max()),
            'cv': round(ptick_arr.std() / (mean_pt + 1e-8), 4),
            'zero_frac': round((ptick_arr == 0).mean(), 4),
            'is_variable': bool(ptick_arr.std() > 0.01),
        })

    edges_df = pd.DataFrame(records)
    edges_df = edges_df.sort_values(['category', 'func_id', 'from_pc', 'to_pc'])
    edges_df.to_csv(OUT_DIR / "all_edges_inventory.csv", index=False)

    for cat in ['controller', 'math', 'library']:
        sub = edges_df[edges_df['category'] == cat]
        var = sub[sub['is_variable']]
        print(f"  {cat}: {len(sub)} edges ({len(var)} variable)")

    total_mean = edges_df['mean_per_tick'].sum()
    ctrl_math = edges_df[edges_df['category'].isin(['controller', 'math'])]
    ctrl_mean = ctrl_math['mean_per_tick'].sum()
    print(f"\n  Mean branches/tick: {total_mean:.0f}")
    print(f"  Controller+math branches/tick: {ctrl_mean:.0f} ({100*ctrl_mean/total_mean:.1f}%)")

    # ---- Step 5: Write reference files ----
    print("\n=== Step 5: Writing reference files ===")

    # Controller edges
    controller_edges = edges_df[edges_df['category'].isin(['controller', 'math'])]
    controller_edges = controller_edges.sort_values(['func_name', 'from_pc', 'to_pc'])

    with open(OUT_DIR / "controller_edges.txt", "w") as f:
        f.write(f"# Controller + Math Edge Reference ({len(controller_edges)} edges)\n")
        f.write(f"# Generated from {sessions_processed} sessions\n")
        f.write(f"# * = variable edge (std > 0.01)\n")
        f.write(f"# {'='*100}\n")

        cur_func = None
        for _, r in controller_edges.iterrows():
            if r['func_name'] != cur_func:
                cur_func = r['func_name']
                fe = controller_edges[controller_edges['func_name'] == cur_func]
                f.write(f"\n## {cur_func} (func_id={r['func_id']}, {len(fe)} edges)\n")
            var = "*" if r['is_variable'] else " "
            f.write(f"  {var} {r['edge_id']:20s} | "
                    f"mean={r['mean_per_tick']:7.2f} | "
                    f"std={r['std_per_tick']:6.2f} | "
                    f"CV={r['cv']:5.2f} | "
                    f"range=[{r['min_per_tick']},{r['max_per_tick']}] | "
                    f"sessions={r['n_sessions']}/{sessions_processed} | "
                    f"zero={r['zero_frac']:.1%}\n")

        variable_ctrl = controller_edges[controller_edges['is_variable']]
        f.write(f"\n# Summary:\n")
        f.write(f"# Total controller+math edges: {len(controller_edges)}\n")
        f.write(f"# Variable edges (std > 0.01): {len(variable_ctrl)}\n")
        f.write(f"# Constant edges: {len(controller_edges) - len(variable_ctrl)}\n")
        f.write(f"# Mean controller+math branches per tick: {controller_edges['mean_per_tick'].sum():.0f}\n")

    # Library edges
    library_edges = edges_df[edges_df['category'] == 'library']
    library_edges = library_edges.sort_values(['func_name', 'total_count'], ascending=[True, False])

    with open(OUT_DIR / "library_edges.txt", "w") as f:
        lib_pct = 100 - 100 * ctrl_mean / total_mean
        f.write(f"# Library Edge Reference ({len(library_edges)} edges)\n")
        f.write(f"# Generated from {sessions_processed} sessions\n")
        f.write(f"# These account for ~{lib_pct:.0f}% of branch executions per tick\n")
        f.write(f"# * = variable edge (std > 0.01)\n")
        f.write(f"# {'='*100}\n")

        cur_func = None
        for _, r in library_edges.iterrows():
            if r['func_name'] != cur_func:
                cur_func = r['func_name']
                fe = library_edges[library_edges['func_name'] == cur_func]
                f.write(f"\n## {cur_func} (func_id={r['func_id']}, {len(fe)} edges)\n")
            var = "*" if r['is_variable'] else " "
            f.write(f"  {var} {r['edge_id']:20s} | "
                    f"mean={r['mean_per_tick']:7.2f} | "
                    f"std={r['std_per_tick']:6.2f} | "
                    f"CV={r['cv']:5.2f} | "
                    f"range=[{r['min_per_tick']},{r['max_per_tick']}] | "
                    f"sessions={r['n_sessions']}/{sessions_processed}\n")

        f.write(f"\n# Summary:\n")
        f.write(f"# Total library edges: {len(library_edges)}\n")
        f.write(f"# Mean library branches per tick: {library_edges['mean_per_tick'].sum():.0f}\n")

    # JSON versions
    controller_edges.to_json(OUT_DIR / "controller_edges.json", orient="records", indent=2)
    library_edges.to_json(OUT_DIR / "library_edges.json", orient="records", indent=2)

    print(f"  controller_edges.txt: {len(controller_edges)} edges")
    print(f"  library_edges.txt: {len(library_edges)} edges")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("KEY NUMBERS")
    print(f"{'='*70}")
    print(f"Total sessions: {len(manifest)}")
    print(f"Sessions with processed_trace.csv: {manifest['has_processed_trace'].sum()}")
    print(f"Sessions with raw .aspk: {manifest['has_raw_aspk'].sum()}")
    print(f"Total ticks (processed_trace rows): {manifest['processed_trace_rows'].sum():,.0f}")
    print(f"Total unique edges: {len(edges_df)}")
    print(f"Controller+math edges: {len(controller_edges)} ({len(controller_edges[controller_edges['is_variable']])} variable)")
    print(f"Library edges: {len(library_edges)}")
    print(f"Mean branches/tick: {total_mean:.0f}")
    print(f"Mean CONTROLLER+MATH branches/tick: {ctrl_mean:.0f} ({100*ctrl_mean/total_mean:.1f}%)")
    print(f"Same Wasm build: {len(layout_hashes) == 1}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
