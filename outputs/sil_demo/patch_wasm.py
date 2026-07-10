"""patch_wasm.py -- layout-preserving binary mutations for the fault corpus.

Injects a software fault into the shipped WebAssembly controller as an
in-place binary edit that does NOT shift any byte offset, so the program
counters (and therefore the reconstructed-trace vocabulary) are unchanged.
Two mechanism classes (see fault_taxonomy_design.md):

  * constant  -- overwrite a numeric-constant immediate with a same-width
                 replacement (e.g. balanced_counter 1500 -> 150). CWE-682.
  * opcode    -- swap a single-byte relational/arithmetic opcode
                 (e.g. f64.lt<->f64.le, f64.add<->f64.sub). CWE-697 / 480.
                 NOTE: a safe opcode swap must target a verified code-section
                 offset (a raw byte scan hits data-byte false positives), so
                 opcode mutants take an explicit offset from the layout, not
                 a search. Provided as an API; corpus offsets are filled from
                 layout.json target sites.

Each mutation verifies the expected old bytes at the offset before writing,
so a wrong offset fails loudly rather than corrupting the binary silently.
"""
import argparse
import shutil
from pathlib import Path

# sLEB128 encodings for the balanced_counter case study (both 3 bytes with the
# 0x41 i32.const opcode: 0x41 <2-byte sLEB128>).
I32_CONST = 0x41
SLEB_1500 = bytes([0xDC, 0x0B])   # 1500
SLEB_150 = bytes([0x96, 0x01])    # 150

# Single-byte f64 opcode pairs for opcode-swap mutants.
OPCODE_SWAP = {
    "f64.lt->le": (0x63, 0x65), "f64.le->lt": (0x65, 0x63),
    "f64.gt->ge": (0x64, 0x66), "f64.ge->gt": (0x66, 0x64),
    "f64.add->sub": (0xA0, 0xA1), "f64.sub->add": (0xA1, 0xA0),
    "f64.mul->div": (0xA2, 0xA3),
}


def patch_bytes(data: bytearray, offset: int, old: bytes, new: bytes):
    """Verify `old` at `offset`, overwrite with equal-length `new`."""
    assert len(old) == len(new), "layout-preserving edits must be same width"
    got = bytes(data[offset:offset + len(old)])
    if got != old:
        raise ValueError(f"offset {offset}: expected {old.hex()} got {got.hex()}")
    data[offset:offset + len(new)] = new


def apply_constant(in_path, out_path, offset, old_val_bytes, new_val_bytes,
                   opcode=I32_CONST):
    """Overwrite a constant immediate (opcode byte + immediate) in place."""
    data = bytearray(Path(in_path).read_bytes())
    patch_bytes(data, offset, bytes([opcode]) + old_val_bytes,
                bytes([opcode]) + new_val_bytes)
    Path(out_path).write_bytes(data)


def apply_opcode(in_path, out_path, offset, name):
    """Swap a single-byte opcode at a verified code-section offset."""
    old, new = OPCODE_SWAP[name]
    data = bytearray(Path(in_path).read_bytes())
    patch_bytes(data, offset, bytes([old]), bytes([new]))
    Path(out_path).write_bytes(data)


# Case-study fault: balanced_counter threshold 1500 -> 150 (stays_balanced),
# the headline constant-level mutation; offset verified unique in module.wasm.
BALANCED_COUNTER_OFFSET = 1462


def seed_balanced_counter_bug(in_path, out_path):
    apply_constant(in_path, out_path, BALANCED_COUNTER_OFFSET,
                   SLEB_1500, SLEB_150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("out")
    ap.add_argument("--bug", default="balanced_counter",
                    help="balanced_counter | const:OFF:OLDHEX:NEWHEX | "
                         "opcode:OFF:NAME")
    args = ap.parse_args()
    if args.bug == "balanced_counter":
        seed_balanced_counter_bug(args.module, args.out)
    elif args.bug.startswith("const:"):
        _, off, oldh, newh = args.bug.split(":")
        apply_constant(args.module, args.out, int(off),
                       bytes.fromhex(oldh), bytes.fromhex(newh))
    elif args.bug.startswith("opcode:"):
        _, off, name = args.bug.split(":")
        apply_opcode(args.module, args.out, int(off), name)
    else:
        raise SystemExit(f"unknown bug spec {args.bug}")
    print(f"[patch] {args.bug} -> {args.out}")


if __name__ == "__main__":
    main()
