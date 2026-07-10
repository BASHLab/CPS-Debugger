#!/usr/bin/env python3
"""Extract all control-flow edges from a Wasm binary (orig_module.wasm).

Parses the original (uninstrumented) WebAssembly binary to build a complete
control-flow graph. PCs match layout.json convention: 1-indexed byte offsets
from the start of each function's instruction stream.

Output: auto_token_mapping.json with all (func_id, from_pc, to_pc) edges.
With --prune: also outputs pruned_token_mapping.json excluding standard
library functions (printf, malloc, stdio, etc.) identified by naming patterns.

Usage:
    python3 extract_wasm_edges.py [--wasm PATH] [--layout PATH] [--verify-traces]
    python3 extract_wasm_edges.py --wasm PATH --layout PATH --output OUT --prune
"""
import argparse, json, re, sys
from pathlib import Path


# ── LEB128 ────────────────────────────────────────────────────────────
def read_uleb128(data, offset):
    result = 0; shift = 0
    while True:
        byte = data[offset]; offset += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0: break
        shift += 7
    return result, offset

def read_sleb128(data, offset):
    result = 0; shift = 0
    while True:
        byte = data[offset]; offset += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if (byte & 0x80) == 0:
            if shift < 64 and (byte & 0x40):
                result |= -(1 << shift)
            break
    return result, offset

def leb128_size(data, offset):
    """Return number of bytes for a LEB128 value starting at offset."""
    n = 0
    while True:
        n += 1
        if (data[offset + n - 1] & 0x80) == 0:
            return n


# ── Wasm section parsing ─────────────────────────────────────────────
def parse_sections(data):
    assert data[:4] == b'\x00asm' and data[4:8] == b'\x01\x00\x00\x00'
    offset = 8
    sections = {}
    while offset < len(data):
        sec_id = data[offset]; offset += 1
        sec_size, offset = read_uleb128(data, offset)
        sections[sec_id] = data[offset:offset+sec_size]
        offset += sec_size
    return sections

def count_func_imports(import_data):
    offset = 0
    count, offset = read_uleb128(import_data, offset)
    n = 0
    for _ in range(count):
        mod_len, offset = read_uleb128(import_data, offset)
        offset += mod_len
        field_len, offset = read_uleb128(import_data, offset)
        offset += field_len
        kind = import_data[offset]; offset += 1
        if kind == 0:  # func
            _, offset = read_uleb128(import_data, offset)
            n += 1
        elif kind == 1:  # table
            offset += 1  # elem type
            flag, offset = read_uleb128(import_data, offset)
            _, offset = read_uleb128(import_data, offset)
            if flag == 1: _, offset = read_uleb128(import_data, offset)
        elif kind == 2:  # memory
            flag, offset = read_uleb128(import_data, offset)
            _, offset = read_uleb128(import_data, offset)
            if flag == 1: _, offset = read_uleb128(import_data, offset)
        elif kind == 3:  # global
            offset += 2
    return n


# ── Instruction stream decoder ────────────────────────────────────────
# Returns the size (in bytes) of the instruction at data[offset]
def instruction_size(data, offset):
    """Return total byte size of instruction at data[offset]."""
    op = data[offset]

    # block/loop/if: opcode + block_type (1 byte for 0x40/value types)
    if op in (0x02, 0x03, 0x04):
        bt = data[offset + 1]
        if bt == 0x40 or (0x7C <= bt <= 0x7F):
            return 2
        else:
            return 1 + leb128_size(data, offset + 1)  # s33 type index

    # else, end, unreachable, nop, return, drop, select
    if op in (0x00, 0x01, 0x05, 0x0B, 0x0F, 0x1A, 0x1B):
        return 1

    # br, br_if: opcode + label_idx (LEB128)
    if op in (0x0C, 0x0D):
        return 1 + leb128_size(data, offset + 1)

    # br_table: opcode + count (LEB128) + (count+1) labels (each LEB128)
    if op == 0x0E:
        pos = offset + 1
        n_labels, pos = read_uleb128(data, pos)
        for _ in range(n_labels + 1):
            _, pos = read_uleb128(data, pos)
        return pos - offset

    # call: opcode + func_idx (LEB128)
    if op == 0x10:
        return 1 + leb128_size(data, offset + 1)

    # call_indirect: opcode + type_idx (LEB128) + table_idx (LEB128)
    if op == 0x11:
        s1 = leb128_size(data, offset + 1)
        s2 = leb128_size(data, offset + 1 + s1)
        return 1 + s1 + s2

    # local.get/set/tee, global.get/set: opcode + idx (LEB128)
    if 0x20 <= op <= 0x24:
        return 1 + leb128_size(data, offset + 1)

    # memory load/store: opcode + align (LEB128) + offset (LEB128)
    if 0x28 <= op <= 0x3E:
        s1 = leb128_size(data, offset + 1)
        s2 = leb128_size(data, offset + 1 + s1)
        return 1 + s1 + s2

    # memory.size, memory.grow: opcode + 0x00
    if op in (0x3F, 0x40):
        return 2

    # i32.const: opcode + i32 (sLEB128)
    if op == 0x41:
        return 1 + leb128_size(data, offset + 1)

    # i64.const: opcode + i64 (sLEB128)
    if op == 0x42:
        return 1 + leb128_size(data, offset + 1)

    # f32.const: opcode + 4 bytes
    if op == 0x43:
        return 5

    # f64.const: opcode + 8 bytes
    if op == 0x44:
        return 9

    # 0xFC prefix (saturating truncation, bulk memory, etc.)
    if op == 0xFC:
        pos = offset + 1
        sub_op, pos = read_uleb128(data, pos)
        base = pos - offset
        if sub_op <= 7:  # i32/i64.trunc_sat — no extra immediates
            return base
        elif sub_op == 8:  # memory.init: segment_idx + 0x00
            return base + leb128_size(data, pos) + 1
        elif sub_op == 9:  # data.drop: segment_idx
            return base + leb128_size(data, pos)
        elif sub_op == 10:  # memory.copy: 0x00 0x00
            return base + 2
        elif sub_op == 11:  # memory.fill: 0x00
            return base + 1
        elif sub_op == 12:  # table.init
            return base + leb128_size(data, pos) + leb128_size(data, pos + leb128_size(data, pos))
        elif sub_op == 13:  # elem.drop
            return base + leb128_size(data, pos)
        elif sub_op == 14:  # table.copy
            return base + leb128_size(data, pos) + leb128_size(data, pos + leb128_size(data, pos))
        elif sub_op in (15, 16, 17):  # table.grow/size/fill
            return base + leb128_size(data, pos)
        return base  # fallback

    # 0xFD prefix (SIMD) — unlikely in our binary but handle
    if op == 0xFD:
        pos = offset + 1
        sub_op, pos = read_uleb128(data, pos)
        base = pos - offset
        if sub_op == 12:  # v128.const: 16 bytes
            return base + 16
        if sub_op <= 11:  # v128.load/store: memarg
            s1 = leb128_size(data, pos)
            s2 = leb128_size(data, pos + s1)
            return base + s1 + s2
        return base

    # All other opcodes (arithmetic, comparison, conversion): 1 byte
    return 1


def extract_function_edges(body_bytes, func_id):
    """Parse function body, extract all control-flow edges.

    PCs are 1-indexed byte offsets from instruction stream start
    (matching layout.json convention).

    Returns list of (from_pc, to_pc, edge_type).
    """
    # Skip local declarations
    offset = 0
    n_decls, offset = read_uleb128(body_bytes, offset)
    for _ in range(n_decls):
        _, offset = read_uleb128(body_bytes, offset)  # count
        offset += 1  # type
    inst_start = offset

    # First pass: decode all instructions with their PCs
    instructions = []  # (pc_1indexed, opcode, immediates)
    pos = inst_start
    while pos < len(body_bytes):
        raw_offset = pos - inst_start  # 0-indexed byte offset
        pc = raw_offset + 1            # 1-indexed (layout.json convention)
        op = body_bytes[pos]
        size = instruction_size(body_bytes, pos)

        imm = {}
        if op in (0x0C, 0x0D):  # br, br_if
            label, _ = read_uleb128(body_bytes, pos + 1)
            imm['label'] = label
        elif op == 0x0E:  # br_table
            p = pos + 1
            n_labels, p = read_uleb128(body_bytes, p)
            labels = []
            for _ in range(n_labels + 1):
                lbl, p = read_uleb128(body_bytes, p)
                labels.append(lbl)
            imm['labels'] = labels
        elif op == 0x10:  # call
            idx, _ = read_uleb128(body_bytes, pos + 1)
            imm['func_idx'] = idx
        elif op == 0x11:  # call_indirect
            s1 = leb128_size(body_bytes, pos + 1)
            type_idx, _ = read_uleb128(body_bytes, pos + 1)
            table_idx, _ = read_uleb128(body_bytes, pos + 1 + s1)
            imm['type_idx'] = type_idx
            imm['table_idx'] = table_idx

        instructions.append((pc, op, imm, size))
        pos += size

    # Second pass: match block/loop/if/else/end pairs
    stack = []  # (type_str, inst_index)
    block_info = {}  # inst_index → {type, end_idx, else_idx}

    for i, (pc, op, imm, size) in enumerate(instructions):
        if op == 0x02:  # block
            stack.append(('block', i))
        elif op == 0x03:  # loop
            stack.append(('loop', i))
        elif op == 0x04:  # if
            stack.append(('if', i))
        elif op == 0x05:  # else
            if stack and stack[-1][0] == 'if':
                _, if_idx = stack.pop()
                block_info[if_idx] = {'type': 'if', 'else_idx': i, 'end_idx': None}
                stack.append(('else', if_idx, i))
        elif op == 0x0B:  # end
            if stack:
                top = stack.pop()
                if top[0] == 'block':
                    block_info[top[1]] = {'type': 'block', 'end_idx': i}
                elif top[0] == 'loop':
                    block_info[top[1]] = {'type': 'loop', 'end_idx': i}
                elif top[0] == 'if':
                    block_info[top[1]] = {'type': 'if', 'else_idx': None, 'end_idx': i}
                elif top[0] == 'else':
                    _, if_idx, else_idx = top
                    if if_idx in block_info:
                        block_info[if_idx]['end_idx'] = i
                    block_info[else_idx] = {'type': 'else', 'if_idx': if_idx, 'end_idx': i}

    # Third pass: extract edges by resolving branch targets
    label_stack = [{'type': 'func', 'inst_idx': -1}]
    edges = []

    for i, (pc, op, imm, size) in enumerate(instructions):
        if op == 0x02:  # block
            label_stack.append({'type': 'block', 'inst_idx': i})
        elif op == 0x03:  # loop
            label_stack.append({'type': 'loop', 'inst_idx': i})
        elif op == 0x04:  # if
            label_stack.append({'type': 'if', 'inst_idx': i})
        elif op == 0x05:  # else
            if label_stack and label_stack[-1]['type'] == 'if':
                label_stack.pop()
                label_stack.append({'type': 'else', 'inst_idx': i})
        elif op == 0x0B:  # end
            if label_stack:
                label_stack.pop()

        elif op == 0x0C:  # br
            target_pc = _resolve_label(label_stack, imm['label'],
                                        instructions, block_info)
            if target_pc is not None:
                edges.append((pc, target_pc, 'br'))

        elif op == 0x0D:  # br_if
            target_pc = _resolve_label(label_stack, imm['label'],
                                        instructions, block_info)
            if target_pc is not None:
                edges.append((pc, target_pc, 'br_if_taken'))
            # Fall-through: next instruction
            if i + 1 < len(instructions):
                edges.append((pc, instructions[i+1][0], 'br_if_fallthrough'))

        elif op == 0x0E:  # br_table
            seen = set()
            for lbl in imm['labels']:
                target_pc = _resolve_label(label_stack, lbl,
                                            instructions, block_info)
                if target_pc is not None and target_pc not in seen:
                    edges.append((pc, target_pc, 'br_table'))
                    seen.add(target_pc)

    return edges


def _resolve_label(label_stack, label_idx, instructions, block_info):
    """Resolve br label to target PC (1-indexed)."""
    if label_idx >= len(label_stack):
        return None
    target = label_stack[-(label_idx + 1)]
    ttype = target['type']
    tidx = target['inst_idx']

    if ttype == 'func':
        return None  # function return

    if ttype == 'loop':
        # Loop: branch targets the loop opcode itself
        # (tracer records the loop instruction's PC as the landing point)
        return instructions[tidx][0]

    # block/if/else: branch targets the END instruction itself
    # (tracer records the end opcode's PC as the landing point)
    ref_idx = tidx
    if ttype == 'else':
        pass  # else has its own end_idx in block_info

    bi = block_info.get(ref_idx)
    if bi is None:
        return None
    end_idx = bi.get('end_idx')
    if end_idx is None:
        return None
    return instructions[end_idx][0]  # PC of the END instruction


# ── Main extraction ───────────────────────────────────────────────────
def extract_all_edges(wasm_path, layout_path=None):
    data = Path(wasm_path).read_bytes()
    sections = parse_sections(data)

    n_imports = count_func_imports(sections[2]) if 2 in sections else 0
    print(f"  Imported functions: {n_imports}")

    func_names = {}
    if layout_path:
        layout = json.loads(Path(layout_path).read_text())
        for k, v in layout.items():
            if k.isdigit() and isinstance(v, dict) and 'name' in v:
                func_names[int(k)] = v['name']

    code_data = sections[10]
    offset = 0
    n_funcs, offset = read_uleb128(code_data, offset)
    print(f"  Code section: {n_funcs} function bodies")

    edges_by_func = {}
    total_edges = 0

    for i in range(n_funcs):
        func_id = n_imports + i
        body_size, offset = read_uleb128(code_data, offset)
        body_bytes = code_data[offset:offset+body_size]
        offset += body_size

        try:
            edges = extract_function_edges(body_bytes, func_id)
            edges_by_func[func_id] = edges
            total_edges += len(edges)
        except Exception as e:
            fname = func_names.get(func_id, '?')
            print(f"  WARNING: func {func_id} ({fname}): {e}")
            edges_by_func[func_id] = []

    print(f"  Total edges extracted: {total_edges}")
    return edges_by_func, func_names


def build_token_mapping(edges_by_func):
    all_triples = set()
    for func_id, edges in edges_by_func.items():
        for from_pc, to_pc, etype in edges:
            all_triples.add((func_id, from_pc, to_pc))

    sorted_triples = sorted(all_triples)
    return {
        "vocab_size": len(sorted_triples),
        "triples": [[f, fp, tp] for f, fp, tp in sorted_triples],
        "triple_to_id": {f"{f}:{fp}:{tp}": i
                         for i, (f, fp, tp) in enumerate(sorted_triples)},
    }


def verify_against_traces(token_mapping, trace_dirs, n_sample=10000):
    """Verify vocabulary against observed ASPK trace files.

    Args:
        token_mapping: dict with 'triples' key
        trace_dirs: list of Path objects pointing to trace/ directories
        n_sample: total number of .aspk files to sample across all dirs
    """
    import numpy as np, os

    triple_set = set(tuple(t) for t in token_mapping['triples'])
    per_dir = max(1, n_sample // len(trace_dirs))

    total = 0; missing = set(); observed = set()
    for trace_dir in trace_dirs:
        trace_dir = Path(trace_dir)
        if not trace_dir.is_dir():
            print(f"  WARNING: trace dir not found: {trace_dir}")
            continue
        files = [e.path for e in os.scandir(trace_dir) if e.name.endswith('.aspk')]
        rng = np.random.RandomState(42)
        sample = rng.choice(files, min(per_dir, len(files)), replace=False)
        for path in sample:
            with open(path, 'rb') as f:
                raw = f.read()
            n = (len(raw) - 8) // 8
            if n <= 0: continue
            arr = np.frombuffer(raw[8:8+n*8], dtype=np.uint8).reshape(n, 8).astype(np.int64)
            to_pc   = arr[:,0] | (arr[:,1]<<8) | (arr[:,2]<<16)
            from_pc = arr[:,3] | (arr[:,4]<<8) | (arr[:,5]<<16)
            func    = arr[:,6] | (arr[:,7]<<8)
            total += n
            for j in range(n):
                t = (int(func[j]), int(from_pc[j]), int(to_pc[j]))
                observed.add(t)
                if t not in triple_set:
                    missing.add(t)

    print(f"\n  Verification: {len(observed)} unique triples from "
          f"{total} entries in {n_sample} files")
    print(f"  In vocabulary: {len(observed) - len(missing)}")
    print(f"  MISSING from vocabulary: {len(missing)}")
    if missing:
        for t in sorted(missing)[:30]:
            print(f"    func={t[0]}, from={t[1]}, to={t[2]}")
    n_unobs = len(triple_set - observed)
    print(f"  Vocab entries never observed: {n_unobs}/{len(triple_set)}")
    return missing, observed


def augment_with_observed(token_mapping, missing_triples):
    """Add observed-but-not-static triples (e.g. call_indirect edges) to vocab."""
    all_triples = set(tuple(t) for t in token_mapping['triples'])
    added = 0
    for t in missing_triples:
        if t not in all_triples:
            all_triples.add(t)
            added += 1
    sorted_triples = sorted(all_triples)
    return {
        "vocab_size": len(sorted_triples),
        "triples": [[f, fp, tp] for f, fp, tp in sorted_triples],
        "triple_to_id": {f"{f}:{fp}:{tp}": i
                         for i, (f, fp, tp) in enumerate(sorted_triples)},
    }, added


# ── Static function classification (PL-knowledge-based pruning) ──────
# Standard C library naming patterns — these are universal across C→Wasm
# compilations. Anything NOT matching is assumed to be application code.

_STDLIB_PATTERNS = [
    # WASI imports and wrappers
    r'^__imported_',
    r'^__wasi_',
    # C runtime
    r'^__wasm_call_',          # ctors/dtors
    r'^__main_void$',
    r'^_Exit$',
    r'^abort$',
    r'^__assert_fail$',
    # Memory management
    r'^(dl)?(malloc|calloc|realloc|free)$',
    r'^sbrk$',
    # String / memory operations
    r'^mem(cpy|set|chr|move|cmp)$',
    r'^str(len|nlen|cpy|ncpy|cat|ncat|cmp|ncmp|chr|rchr|error|str|tok|dup)$',
    # stdio / file I/O
    r'^__stdio_',
    r'^__stdout_',
    r'^__ofl_',
    r'^__towrite$',
    r'^__fwritex$',
    r'^__lseek$',
    r'^__isatty$',
    r'^__lctrans$',
    r'^__clock_gettime$',
    r'^(v?s?n?)(printf|fprintf)',   # matches printf, printf_core, vfprintf, etc.
    r'^(fwrite|fread|fputs|fgets|fclose|fopen)$',
    r'^(close|open|read|write|writev)$',
    # printf internals
    r'^(sn_write|pad|pop_arg|long_double_not_supported)$',
    # Wide char
    r'^(wcrtomb|wctomb|wcsrtombs)$',
    # Misc
    r'^dummy$',
    r'^strerror$',
    r'^strnlen$',
]

_MATH_PATTERNS = [
    r'^(a?sin|a?cos|a?tan)(h|f|l)?$',
    r'^(log|log2|log10|exp|exp2)(f|l)?$',
    r'^(sqrt|cbrt)(f|l)?$',
    r'^f(min|max|abs|mod)(f|l)?$',
    r'^(ceil|floor|round|trunc|rint|nearbyint)(f|l)?$',
    r'^(scalbn|ldexp|frexp|modf|remainder)(f|l)?$',
    r'^(pow|hypot)(f|l)?$',
    r'^__(sin|cos|tan|rem_pio2)',   # internal math helpers
    r'^__math_',                     # __math_divzero, __math_invalid
]

# Compile once
_STDLIB_RE = [re.compile(p) for p in _STDLIB_PATTERNS]
_MATH_RE = [re.compile(p) for p in _MATH_PATTERNS]


def classify_function(name):
    """Classify a function as 'application', 'math', or 'stdlib'.

    Uses standard C library naming conventions — generalizable to any
    C-to-WebAssembly compilation. Unknown names default to 'application'.
    """
    for pat in _MATH_RE:
        if pat.match(name):
            return 'math'
    for pat in _STDLIB_RE:
        if pat.match(name):
            return 'stdlib'
    return 'application'


def classify_all_functions(func_names, edges_by_func):
    """Classify all functions and return {func_id: category}."""
    classification = {}
    for fid in edges_by_func:
        name = func_names.get(fid, f'unknown_{fid}')
        classification[fid] = classify_function(name)
    return classification


def build_pruned_token_mapping(edges_by_func, func_classes, keep_categories):
    """Build token mapping keeping only edges from specified categories."""
    all_triples = set()
    for func_id, edges in edges_by_func.items():
        if func_classes.get(func_id) in keep_categories:
            for from_pc, to_pc, etype in edges:
                all_triples.add((func_id, from_pc, to_pc))

    sorted_triples = sorted(all_triples)
    return {
        "vocab_size": len(sorted_triples),
        "triples": [[f, fp, tp] for f, fp, tp in sorted_triples],
        "triple_to_id": {f"{f}:{fp}:{tp}": i
                         for i, (f, fp, tp) in enumerate(sorted_triples)},
    }


def main():
    ap = argparse.ArgumentParser(
        description="Extract all control-flow edges from a Wasm binary.")
    ap.add_argument("--wasm", required=True,
        help="Path to original (uninstrumented) .wasm binary")
    ap.add_argument("--layout", default=None,
        help="Path to layout.json (optional, for function names)")
    ap.add_argument("--output", required=True,
        help="Output path for auto_token_mapping.json")
    ap.add_argument("--old-vocab", default=None,
        help="Path to old token_mapping.json for comparison (optional)")
    ap.add_argument("--verify-traces", nargs='*', default=None, metavar="TRACE_DIR",
        help="Trace directories to verify against (each containing .aspk files)")
    ap.add_argument("--n-sample", type=int, default=10000,
        help="Number of .aspk files to sample for verification (default: 10000)")
    ap.add_argument("--prune", action="store_true",
        help="Also output a pruned vocabulary excluding standard library functions")
    ap.add_argument("--pruned-output", default=None,
        help="Output path for pruned_token_mapping.json (default: sibling of --output)")
    ap.add_argument("--drop-math", action="store_true",
        help="Also exclude math functions from pruned vocabulary (default: keep)")
    args = ap.parse_args()

    print(f"Extracting Wasm control-flow edges from {args.wasm}...")
    edges_by_func, func_names = extract_all_edges(args.wasm, args.layout)

    print(f"\nEdges per function:")
    for fid in sorted(edges_by_func.keys()):
        edges = edges_by_func[fid]
        fname = func_names.get(fid, '?')
        if edges:
            print(f"  func {fid:>3d} {fname:30s}: {len(edges)} edges")

    tm = build_token_mapping(edges_by_func)
    print(f"\nVocabulary size: {tm['vocab_size']}")

    # Compare with old vocab if provided
    if args.old_vocab:
        old_path = Path(args.old_vocab)
        if old_path.exists():
            old_tm = json.loads(old_path.read_text())
            old_set = set(tuple(t) for t in old_tm['triples'])
            new_set = set(tuple(t) for t in tm['triples'])
            n_match = len(old_set & new_set)
            print(f"\n  Old vocab: {old_tm['vocab_size']}, New vocab: {tm['vocab_size']}")
            print(f"  Old triples found in new: {n_match}/{len(old_set)}")
            if n_match < len(old_set):
                missing = sorted(old_set - new_set)
                print(f"  Missing ({len(missing)}):")
                for t in missing[:10]:
                    print(f"    func={t[0]} ({func_names.get(t[0],'?')}), "
                          f"from={t[1]}, to={t[2]}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tm, indent=2))
    print(f"\nSaved → {out_path}")

    fn_path = out_path.parent / "func_id_to_name.json"
    fn_path.write_text(json.dumps({str(k): v for k, v in func_names.items()}, indent=2))
    print(f"Saved → {fn_path}")

    # ── Static pruning ────────────────────────────────────────────────
    if args.prune:
        if not func_names:
            print("\nERROR: --prune requires --layout (need function names)")
            sys.exit(1)

        func_classes = classify_all_functions(func_names, edges_by_func)

        # Report classification
        by_cat = {}
        for fid, cat in sorted(func_classes.items()):
            by_cat.setdefault(cat, []).append(fid)

        print(f"\n{'='*60}")
        print("Function classification (static, PL-knowledge-based)")
        print(f"{'='*60}")
        for cat in ['application', 'math', 'stdlib']:
            fids = by_cat.get(cat, [])
            n_edges = sum(len(edges_by_func.get(f, [])) for f in fids)
            print(f"\n  {cat}: {len(fids)} functions, {n_edges} static edges")
            for fid in fids:
                fname = func_names.get(fid, '?')
                n_e = len(edges_by_func.get(fid, []))
                print(f"    func {fid:>3d} {fname:30s}: {n_e} edges")

        # Build pruned mapping
        keep = {'application', 'math'}
        if args.drop_math:
            keep = {'application'}
            print(f"\n  --drop-math: excluding math functions")

        pruned_tm = build_pruned_token_mapping(edges_by_func, func_classes, keep)
        print(f"\n  Full vocab:   {tm['vocab_size']} edges")
        print(f"  Pruned vocab: {pruned_tm['vocab_size']} edges "
              f"({100*(1 - pruned_tm['vocab_size']/tm['vocab_size']):.1f}% reduction)")

        # Save pruned mapping
        pruned_path = Path(args.pruned_output) if args.pruned_output else \
            out_path.parent / "pruned_token_mapping.json"
        pruned_path.parent.mkdir(parents=True, exist_ok=True)
        pruned_tm_out = dict(pruned_tm)
        pruned_tm_out["pruning_info"] = {
            "source": str(out_path.name),
            "original_vocab_size": tm['vocab_size'],
            "kept_categories": sorted(keep),
            "functions_kept": {
                str(fid): func_names.get(fid, '?')
                for fid in sorted(func_classes)
                if func_classes[fid] in keep
            },
            "functions_dropped": {
                str(fid): func_names.get(fid, '?')
                for fid in sorted(func_classes)
                if func_classes[fid] not in keep
            },
        }
        pruned_path.write_text(json.dumps(pruned_tm_out, indent=2))
        print(f"  Saved → {pruned_path}")

        # Compare with old vocab if provided
        if args.old_vocab:
            old_path = Path(args.old_vocab)
            if old_path.exists():
                old_tm = json.loads(old_path.read_text())
                old_set = set(tuple(t) for t in old_tm['triples'])
                pruned_set = set(tuple(t) for t in pruned_tm['triples'])
                overlap = len(old_set & pruned_set)
                only_old = old_set - pruned_set
                only_new = pruned_set - old_set
                print(f"\n  Pruned vs old vocab ({old_tm['vocab_size']} edges):")
                print(f"    Overlap: {overlap}")
                print(f"    In old but not pruned: {len(only_old)}")
                print(f"    In pruned but not old: {len(only_new)}")
                if only_old:
                    print(f"    Missing from pruned (first 10):")
                    for t in sorted(only_old)[:10]:
                        print(f"      func={t[0]} ({func_names.get(t[0],'?')}), "
                              f"from={t[1]}, to={t[2]}")

    if args.verify_traces is not None:
        trace_dirs = args.verify_traces
        if not trace_dirs:
            print("ERROR: --verify-traces requires at least one trace directory")
            sys.exit(1)
        print(f"\nVerifying against {len(trace_dirs)} trace directories...")
        missing, observed = verify_against_traces(tm, trace_dirs, args.n_sample)
        if missing:
            print(f"\n  Augmenting vocabulary with {len(missing)} "
                  f"observed-but-not-static triples (call_indirect edges)...")
            tm, n_added = augment_with_observed(tm, missing)
            print(f"  Added {n_added} triples → vocab size: {tm['vocab_size']}")
            out_path.write_text(json.dumps(tm, indent=2))
            print(f"  Re-saved → {out_path}")
            # Re-verify
            missing2, _ = verify_against_traces(tm, trace_dirs, args.n_sample)
            assert len(missing2) == 0, f"Still missing: {missing2}"


if __name__ == "__main__":
    main()
