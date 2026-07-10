"""
phase2_B1_prepare_code_context.py — Extract C++ source for LLM prompts.

Reads key C source files from the tarball, produces a compact code_context.json
suitable for embedding in LLM prompts.

Output: outputs/phase2/code_context.json
"""

import json
import tarfile
from pathlib import Path

ROOT    = Path("/home/simran/allspark-data-exploration/CPS-Debugger")
TARBALL = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs"
               "/2025/2025-03-13_09-23-44.tar.gz")
OUT     = ROOT / "outputs/phase2"
OUT.mkdir(parents=True, exist_ok=True)

KEY_FILES = {
    "pend_monolithic.c": "data/2025-03-13_09-23-44/code/src/pend_mono/pend_monolithic.c",
    "standup_reliable.c": "data/2025-03-13_09-23-44/code/src/pendulum_controller/standup_reliable.c",
    "lqrsim.c":           "data/2025-03-13_09-23-44/code/src/pendulum_controller/lqrsim.c",
    "log_data.c":         "data/2025-03-13_09-23-44/code/src/pend_mono/log_data.c",
    "general.c":          "data/2025-03-13_09-23-44/code/src/pendulum_controller/general.c",
}

SYSTEM_DESCRIPTION = """\
This is an industrial inverted pendulum controlled by a Bosch Rexroth ctrlX CORE system.
The controller is written in C, compiled to WebAssembly (Wasm), and runs in real-time at
1 kHz (one 1ms control cycle per call to tick()).

SYSTEM:
- Cart slides on a horizontal track (range approx -0.18 m to +0.18 m from center)
- Pendulum arm attached at cart center, free to rotate 360°
- Angle = 0 rad: pendulum pointing straight UP (balanced)
- Angle = ±π rad: pendulum hanging straight DOWN
- Angular velocity = positive when rotating toward upright

CONTROLLER STATES (stored in global pendulum_state):
  0 = SWINGUP: Pendulum is down or mid-swing. Cart oscillates to pump energy
      into the pendulum. Calls pou_standup_rel() each tick.
  1 = BALANCE: Pendulum is near upright (|angle| < threshold). LQR controller
      maintains balance and moves cart to target position. Calls pou_lqr_sim().
      Target positions cycle through: 0.0 m, +0.05 m, -0.09 m, +0.09 m
  2 = RESET: Cart has hit track limits. Cart returns to center before resuming.
      tick() handles reset directly (no sub-function call to pou_standup_rel/lqr).

KEY CONTROL FLOW in tick():
  - Reads physical sensors from EtherCAT (cart position, velocity, angle, angular velocity)
  - Checks if cart is at track limit → sets state=RESET
  - Checks if pendulum is near upright → transitions state 0→1 or 1→0
  - Dispatches to pou_standup_rel() [SWINGUP] or pou_lqr_sim() [BALANCE]
  - Logs data via log_data()
  - Writes actuator commands back to EtherCAT

EXECUTION TRACING:
- The Wasm runtime records branch counts per 1ms cycle
- cf_table[func_id][src_pc][dst_pc] = count of times that branch was taken
- Physical sensors determine which code paths execute and how many loop iterations run
- memcpy is heavily used for state vector copying; its branch count encodes data volume
"""

def read_file_from_tar(tf, archive_path):
    try:
        m = tf.getmember(archive_path)
        return tf.extractfile(m).read().decode("utf-8", errors="replace")
    except KeyError:
        return None


def truncate_source(src, max_lines=200):
    """Keep the first max_lines lines; add truncation notice if needed."""
    lines = src.splitlines()
    if len(lines) <= max_lines:
        return src
    return "\n".join(lines[:max_lines]) + f"\n... [truncated at {max_lines} lines]"


def main():
    print(f"Reading source from {TARBALL.name}...")
    sources = {}
    with tarfile.open(TARBALL, "r:gz") as tf:
        for name, archive_path in KEY_FILES.items():
            src = read_file_from_tar(tf, archive_path)
            if src:
                sources[name] = src
                print(f"  {name}: {len(src.splitlines())} lines")
            else:
                print(f"  {name}: NOT FOUND in tarball")

    # Compact versions for LLM prompts (truncated to ~200 lines each)
    compact = {name: truncate_source(src) for name, src in sources.items()}

    # Vocabulary annotations from vocab_info
    vocab_path = ROOT / "outputs/experiments/vocab_info.json"
    with open(vocab_path) as f:
        vocab_info = json.load(f)
    vocab  = vocab_info["vocab"]
    labels = vocab_info.get("labels", {})
    fn_names = vocab_info.get("fn_names", {})

    # Build annotated branch list for LLM context
    branch_annotations = []
    for bkey in vocab:
        parts  = bkey.split(":")
        func_id, src_pc, dst_pc = parts[0], parts[1], parts[2]
        fn_name = fn_names.get(func_id, f"func_{func_id}")
        label   = labels.get(bkey, bkey)
        branch_annotations.append({
            "id": bkey,
            "function": fn_name,
            "label": label,
        })

    ctx = {
        "system_description": SYSTEM_DESCRIPTION,
        "source_files": compact,
        "source_files_full": sources,
        "branch_vocabulary": branch_annotations,
        "function_map": fn_names,
        "state_map": {"0": "SWINGUP", "1": "BALANCE", "2": "RESET"},
        "target_positions": [0.0, 0.05, -0.09, 0.09],
    }

    out_path = OUT / "code_context.json"
    with open(out_path, "w") as f:
        json.dump(ctx, f, indent=2)
    total_chars = sum(len(v) for v in sources.values())
    print(f"\nSaved code_context.json ({total_chars:,} chars of source, "
          f"{len(branch_annotations)} annotated branches)")


if __name__ == "__main__":
    main()
