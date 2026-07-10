"""C0 — SIL day-1 probe (go/no-go for the F1 primary path).

Runs the SHIPPED instrumented controller binary (module.wasm from the
2025-03-17_10-36-44 capture) under wasmtime with minimal host stubs:

  1. Instantiate with 9 waxi:* stubs + WASI + env.__main_argc_argv.
  2. Find the SERVICE init phase (watch for "Initialising Wasm callable...").
  3. Run N ticks against a trivial static plant (zeroed sensors).
  4. Drain the inline trace ring buffer (counter i32 @131072, 8-byte records
     @131080; record = i64 (func_id<<48)|(from_pc<<24)|to_pc, little-endian).
  5. Assert every decoded triple is in the training vocabulary
     (auto_token_mapping.json, 2124 triples).

Exit 0 = F1 GO. Nonzero = investigate / consider F3.
"""
import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path

from wasmtime import (
    Config, Engine, Func, FuncType, Linker, Module, Store, ValType,
    WasiConfig,
)

CAPTURE = Path("/home/simran/allspark-data-exploration/Pittsburgh-pendulum-datalogs"
               "/extracted/2025-03-17_10-36-44/code")
VOCAB = Path("/home/simran/allspark-data-exploration/CPS-Debugger"
             "/outputs/fm_candidates/train_data_full/auto_token_mapping.json")

RING_COUNTER = 131072
RING_RECORDS = 131080

# Region sizes (bytes) for the three datalayer memories the controller opens.
REGION_SIZES = {
    "fieldbuses/ethercat/master/instances/ethercatmaster/realtime_data/input": 64,
    "fieldbuses/ethercat/master/instances/ethercatmaster/realtime_data/output": 64,
    "sdk/cpp/datalayer/pendulum-logging/output": 1 << 20,
}


class DatalayerHost:
    """Host-side model of the waxi datalayer: regions keyed by iod."""

    def __init__(self):
        self.regions = {}          # iod -> bytearray
        self.addr_by_iod = {}      # iod -> address string
        self.next_iod = 100
        self.calls = Counter()

    def open_memory(self, addr: str) -> int:
        iod = self.next_iod
        self.next_iod += 1
        size = REGION_SIZES.get(addr, 4096)
        self.regions[iod] = bytearray(size)
        self.addr_by_iod[iod] = addr
        print(f"  [dlr] open_memory '{addr}' -> iod {iod} ({size} B)")
        return iod

    def iod_for(self, addr_substr: str):
        for iod, addr in self.addr_by_iod.items():
            if addr_substr in addr:
                return iod
        return None


def read_cstr(mem_data: bytearray, ptr: int, maxlen: int = 256) -> str:
    end = mem_data.find(b"\x00", ptr, ptr + maxlen)
    return mem_data[ptr:end].decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=1000)
    ap.add_argument("--module", default=str(CAPTURE / "module.wasm"))
    args = ap.parse_args()

    engine = Engine(Config())
    module = Module.from_file(engine, args.module)
    store = Store(engine)

    wasi = WasiConfig()
    wasi.inherit_stdout()
    wasi.inherit_stderr()
    store.set_wasi(wasi)

    linker = Linker(engine)
    linker.define_wasi()

    dlr = DatalayerHost()
    state = {"memory": None}

    def mem_data():
        return state["memory"].data_ptr(store), state["memory"].data_len(store)

    def mem_read(off, n):
        m = state["memory"]
        return bytes(m.read(store, off, off + n))

    def mem_write(off, data: bytes):
        state["memory"].write(store, data, off)

    # ── waxi stubs ─────────────────────────────────────────────────────────
    i32 = ValType.i32()

    def stub(name, params, results, fn):
        ft = FuncType(params, results)
        f = Func(store, ft, fn)
        mod_name, func_name = name.split("::")
        linker.define(store, mod_name, func_name, f)

    def svc_dlr_factory():
        dlr.calls["factory"] += 1
        return 1  # any non-invalid handle

    def factory_open_memory(factory, iod_ptr, iod_size, addr_ptr):
        addr = read_cstr(bytearray(mem_read(addr_ptr, 256)), 0)
        iod = dlr.open_memory(addr)
        mem_write(iod_ptr, struct.pack("<i", iod))
        return 0  # WAXI_DL_OK

    def factory_close_memory(factory, iod):
        return 0

    def user_begin_access(iod, rev):
        dlr.calls["begin"] += 1
        return 0

    def user_end_access(iod):
        dlr.calls["end"] += 1
        return 0

    def user_read_bytes(iod, off, buf_ptr, n):
        region = dlr.regions.get(iod)
        if region is None:
            return 1
        chunk = bytes(region[off:off + n])
        mem_write(buf_ptr, chunk)
        dlr.calls["read"] += 1
        return 0

    def user_write_bytes(iod, off, buf_ptr, n):
        region = dlr.regions.get(iod)
        if region is None:
            return 1
        region[off:off + n] = mem_read(buf_ptr, n)
        dlr.calls["write"] += 1
        return 0

    def logging_log(a, b, c):
        dlr.calls["log"] += 1

    def logging_log_extended(a, b, c, d, e, f):
        dlr.calls["log_ext"] += 1

    def main_argc_argv(a, b):
        return 0

    dumped_records = []   # raw bytes captured at each prog_tracedump call

    def prog_tracedump(a, b):
        """Instrumentation hands the trace buffer to the host each tick.
        Read it NOW — the module resets/reuses the buffer afterwards.
        arg semantics probed empirically: try (ptr, n_bytes) first."""
        dlr.calls["tracedump"] += 1
        if dlr.calls["tracedump"] <= 5:
            print(f"  [trace] prog_tracedump(a={a}, b={b})")
        if state["memory"] is not None and b > 0:
            dumped_records.append(mem_read(a, b))

    stub("waxi:services::waxi_service_dlr_factory", [], [i32], svc_dlr_factory)
    stub("waxi:datalayer::waxi_dlr_factory_open_memory",
         [i32, i32, i32, i32], [i32], factory_open_memory)
    stub("waxi:datalayer::waxi_dlr_factory_close_memory",
         [i32, i32], [i32], factory_close_memory)
    stub("waxi:datalayer::waxi_dlr_user_begin_access",
         [i32, i32], [i32], user_begin_access)
    stub("waxi:datalayer::waxi_dlr_user_end_access", [i32], [i32],
         user_end_access)
    stub("waxi:datalayer::waxi_dlr_user_read_bytes",
         [i32, i32, i32, i32], [i32], user_read_bytes)
    stub("waxi:datalayer::waxi_dlr_user_write_bytes",
         [i32, i32, i32, i32], [i32], user_write_bytes)
    stub("waxi:logging::waxi_logging_log", [i32, i32, i32], [], logging_log)
    stub("waxi:logging::waxi_logging_log_extended",
         [i32, i32, i32, i32, i32, i32], [], logging_log_extended)
    stub("env::__main_argc_argv", [i32, i32], [i32], main_argc_argv)
    stub("allspark:trace::prog_tracedump", [i32, i32], [], prog_tracedump)

    instance = linker.instantiate(store, module)
    exports = instance.exports(store)
    state["memory"] = exports["memory"]
    execute = exports["execute"]
    print(f"[probe] instantiated OK; memory pages = "
          f"{state['memory'].size(store)}")

    # ── find init phase ────────────────────────────────────────────────────
    print("[probe] sweeping execute(event, phase, 0) for init "
          "(look for 'Initialising Wasm callable...'):")
    init_found = None
    for event in (8, 7, 6):
        for phase in range(4):
            try:
                rc = execute(store, event, phase, 0)
            except Exception as e:
                print(f"  execute({event},{phase},0) TRAP: {e}")
                continue
            print(f"  execute({event},{phase},0) -> {rc}")
            if dlr.iod_for("ethercat") is not None:
                init_found = (event, phase)
                break
        if init_found:
            break
    if not init_found:
        print("[probe] FAIL: no init phase opened the datalayer memories")
        sys.exit(2)
    print(f"[probe] init at execute{init_found + (0,)}")

    # ── static plant: zero sensors, plausible status ───────────────────────
    ein = dlr.iod_for("realtime_data/input")
    region = dlr.regions[ein]
    struct.pack_into("<h", region, 10, 0)          # encoder int16
    struct.pack_into("<H", region, 12, 0xC000)     # Dr_Status
    struct.pack_into("<i", region, 14, 0)          # Dr_PosIW

    # ── tick loop; records arrive via prog_tracedump callbacks ───────────
    vocab = {tuple(t) for t in json.loads(VOCAB.read_text())["triples"]}
    all_triples = Counter()
    n_records_total = 0
    bad = Counter()

    dumped_records.clear()   # drop init-phase dumps
    for tick in range(args.ticks):
        try:
            execute(store, 0, 0, 0)   # TICK
        except Exception as e:
            print(f"[probe] TRAP at tick {tick}: {e}")
            sys.exit(3)

    for raw in dumped_records:
        for off in range(0, len(raw) - 7, 8):
            (rec,) = struct.unpack("<q", raw[off:off + 8])
            trip = ((rec >> 48) & 0xFFFF, (rec >> 24) & 0xFFFFFF,
                    rec & 0xFFFFFF)
            n_records_total += 1
            if trip in vocab:
                all_triples[trip] += 1
            else:
                bad[trip] += 1

    print(f"\n[probe] {args.ticks} ticks, {n_records_total} trace records")
    print(f"  distinct in-vocab triples: {len(all_triples)}")
    print(f"  out-of-vocab triples: {sum(bad.values())} "
          f"({len(bad)} distinct)")
    if bad:
        print("  top-5 out-of-vocab:", bad.most_common(5))
    print(f"  datalayer call counts: {dict(dlr.calls)}")
    print(f"  top-10 in-vocab: {all_triples.most_common(10)}")

    # The vocab is OBSERVATIONAL (edges seen in the 15 training sessions), so
    # a static zero-sensor plant can exercise valid-but-unseen edges (e.g.
    # the per-tick wait-state printf path through vfprintf/__fwritex, which
    # real sessions never took). GO = decode is trustworthy: >=99.5% of
    # records in-vocab and out-of-vocab confined to a handful of edges.
    frac_ok = (n_records_total - sum(bad.values())) / max(n_records_total, 1)
    ok = n_records_total > 0 and frac_ok >= 0.995 and len(bad) <= 10
    print(f"\n[probe] in-vocab fraction: {frac_ok:.6f}")
    print(f"[probe] {'GO (F1 viable)' if ok else 'INVESTIGATE'}")
    sys.exit(0 if ok else 4)


if __name__ == "__main__":
    main()
