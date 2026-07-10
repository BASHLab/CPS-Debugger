#!/usr/bin/env python3
"""Probe .aspk binary format by cross-referencing with processed_trace counts."""
import json, struct, sys, csv
from collections import Counter
from pathlib import Path

SESSION = "2025-03-18_12-39-10"
EXTRACTED = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs/extracted")
SESS = EXTRACTED / SESSION
TRACE_DIR = SESS / "trace"
PT_PATH = SESS / "processed_trace" / "processed_trace.csv"

def parse_cf_table(s):
    cf = json.loads(s)
    out = Counter()
    for fid_s, fd in cf.items():
        fid = int(fid_s)
        for fp_s, td in fd.items():
            fp = int(fp_s)
            for tp_s, c in td.items():
                out[(fid, fp, int(tp_s))] = c
    return out

def list_aspk_for_ts_range(ts_min_us, ts_max_us, max_files=20):
    """Scan trace dir for files whose ts is within range."""
    out = []
    for f in TRACE_DIR.iterdir():
        name = f.name
        if not name.endswith(".aspk"): continue
        try:
            ts = int(name.split("_ts:")[1].rsplit(".",1)[0])
        except Exception:
            continue
        if ts_min_us <= ts <= ts_max_us:
            out.append((ts, f))
            if len(out) >= max_files: break
    return sorted(out)

def decode_3_3_2(data, skip_header=0):
    """3 bytes target, 3 bytes source, 2 bytes func, little endian."""
    entries = []
    for i in range(skip_header, len(data), 8):
        chunk = data[i:i+8]
        if len(chunk) < 8: break
        target = chunk[0] | (chunk[1] << 8) | (chunk[2] << 16)
        source = chunk[3] | (chunk[4] << 8) | (chunk[5] << 16)
        func   = chunk[6] | (chunk[7] << 8)
        entries.append((func, source, target))
    return entries

def decode_3_3_2_swap(data, skip_header=0):
    """3 bytes source, 3 bytes target, 2 bytes func."""
    entries = []
    for i in range(skip_header, len(data), 8):
        chunk = data[i:i+8]
        if len(chunk) < 8: break
        source = chunk[0] | (chunk[1] << 8) | (chunk[2] << 16)
        target = chunk[3] | (chunk[4] << 8) | (chunk[5] << 16)
        func   = chunk[6] | (chunk[7] << 8)
        entries.append((func, source, target))
    return entries

def main():
    # Step 1: Read first row of processed_trace, get a target timestamp
    with open(PT_PATH) as f:
        reader = csv.DictReader(f)
        # Read a few rows to get a representative one
        rows = []
        for i, row in enumerate(reader):
            if i >= 5: break
            rows.append(row)

    # Use row 0
    target = rows[0]
    target_ts_us = int(target["timestamp"])
    target_cf = parse_cf_table(target["cf_table"])
    target_total = sum(target_cf.values())
    print(f"PT[0] ts={target_ts_us}  total_edges={target_total}  unique_triples={len(target_cf)}")
    print(f"  sample triples: {list(target_cf.items())[:5]}")

    # Step 2: Find an aspk file with that exact ts (or closest)
    # ts in filename is in us. Let's find the aspk with matching ts.
    candidates = list_aspk_for_ts_range(target_ts_us - 100, target_ts_us + 100, max_files=20)
    print(f"\nFound {len(candidates)} candidate aspk files in ±100µs window:")
    for ts, p in candidates[:5]:
        print(f"  ts={ts} (delta={ts-target_ts_us:+d})  size={p.stat().st_size}  name={p.name}")

    if not candidates:
        # Walk forward in PT until we find a row that aligns
        print("No match in PT[0]. Searching for any matching row...")
        return

    # Step 3: Decode the matching aspk file with both hypotheses, with and without header
    ts, fp = candidates[0]
    data = fp.read_bytes()
    print(f"\nDecoding {fp.name} ({len(data)} bytes = {len(data)//8} 8-byte chunks)")

    for skip in [0, 8]:
        for layout, fn in [("(target,source)", decode_3_3_2), ("(source,target)swap", decode_3_3_2_swap)]:
            entries = fn(data, skip_header=skip)
            cnt = Counter(entries)
            n_match = sum(1 for tr, c in cnt.items() if target_cf.get(tr, 0) == c)
            n_total = len(cnt)
            tot_edges = sum(cnt.values())
            print(f"  layout={layout:25s} skip={skip}  decoded {len(entries)} entries, {n_total} unique  -> {n_match}/{n_total} matching counts (sum={tot_edges} vs {target_total})")
            if n_match > 0:
                # show first 3 matched and unmatched
                matched = [(tr,c) for tr,c in cnt.items() if target_cf.get(tr,0)==c][:3]
                unmatched = [(tr,c) for tr,c in cnt.items() if target_cf.get(tr,0)!=c][:3]
                print(f"    matched examples: {matched}")
                print(f"    unmatched examples: {unmatched}")

if __name__ == "__main__":
    main()
