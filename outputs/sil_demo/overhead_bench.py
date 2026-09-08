"""F2' — first-party overhead of the trace instrumentation.

Runs the deployed INSTRUMENTED controller binary (module.wasm) and its
UNINSTRUMENTED original (orig_module.wasm) on identical replayed sensor
inputs under wasmtime, timing each 1 kHz tick. The ratio is a first-party
measurement of what our bytecode-level branch tracing costs, on the same
code, same inputs, same host.

Caveat for the paper: a desktop JIT runtime, not the ctrlX real-time
runtime; relative cost is the claim, absolute times are not.

Inputs are replayed from a real capture: pendulum angle is converted back
to 13-bit encoder counts and cart position to the drive's integer position
word, so the controller executes realistic paths.

Output: outputs/sil_demo/overhead_results.json
"""
import os as _os
from pathlib import Path as _Path

# Roots. Override with environment variables; defaults assume this checkout and
# the dataset release unpacked as CPSD_DATA (see outputs/sil_demo/README.md).
REPO = _Path(_os.environ.get("CPSD_REPO", _Path(__file__).resolve().parents[2]))
CPSD_DATA = _Path(_os.environ.get("CPSD_DATA", REPO / "data"))
CPSD_CAPTURE = _Path(_os.environ.get("CPSD_CAPTURE", CPSD_DATA / "captures"))

import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np

from wasmtime import Config, Engine, Func, FuncType, Linker, Module, Store, \
    ValType, WasiConfig

CAPTURE = CPSD_CAPTURE / "2025-03-17_10-36-44" / "code"
DATA = CPSD_DATA
SESSION = "2025-03-25_13-39-06"

REGION_SIZES = {
    "fieldbuses/ethercat/master/instances/ethercatmaster/realtime_data/input": 64,
    "fieldbuses/ethercat/master/instances/ethercatmaster/realtime_data/output": 64,
    "sdk/cpp/datalayer/pendulum-logging/output": 1 << 20,
}


def build_instance(engine, path: Path):
    module = Module.from_file(engine, str(path))
    store = Store(engine)
    wasi = WasiConfig()
    store.set_wasi(wasi)   # stdout suppressed: printf cost excluded for both
    linker = Linker(engine)
    linker.define_wasi()

    state = {"memory": None}
    regions, addr_by_iod, next_iod = {}, {}, [100]

    def mem_read(off, n):
        return bytes(state["memory"].read(store, off, off + n))

    def mem_write(off, data):
        state["memory"].write(store, data, off)

    def read_cstr(ptr):
        raw = mem_read(ptr, 256)
        return raw[:raw.find(b"\x00")].decode("utf-8", "replace")

    i32 = ValType.i32()

    def stub(name, params, results, fn):
        mod, func = name.split("::")
        linker.define(store, mod, func, Func(store, FuncType(params, results), fn))

    stub("waxi:services::waxi_service_dlr_factory", [], [i32], lambda: 1)

    def open_memory(f, iod_ptr, sz, addr_ptr):
        addr = read_cstr(addr_ptr)
        iod = next_iod[0]; next_iod[0] += 1
        regions[iod] = bytearray(REGION_SIZES.get(addr, 4096))
        addr_by_iod[iod] = addr
        mem_write(iod_ptr, struct.pack("<i", iod))
        return 0
    stub("waxi:datalayer::waxi_dlr_factory_open_memory",
         [i32]*4, [i32], open_memory)
    stub("waxi:datalayer::waxi_dlr_factory_close_memory",
         [i32]*2, [i32], lambda f, i: 0)
    stub("waxi:datalayer::waxi_dlr_user_begin_access",
         [i32]*2, [i32], lambda i, r: 0)
    stub("waxi:datalayer::waxi_dlr_user_end_access", [i32], [i32], lambda i: 0)

    def read_bytes(iod, off, buf, n):
        mem_write(buf, bytes(regions[iod][off:off + n])); return 0
    def write_bytes(iod, off, buf, n):
        regions[iod][off:off + n] = mem_read(buf, n); return 0
    stub("waxi:datalayer::waxi_dlr_user_read_bytes", [i32]*4, [i32], read_bytes)
    stub("waxi:datalayer::waxi_dlr_user_write_bytes", [i32]*4, [i32], write_bytes)
    stub("waxi:logging::waxi_logging_log", [i32]*3, [], lambda a, b, c: None)
    stub("waxi:logging::waxi_logging_log_extended", [i32]*6, [],
         lambda a, b, c, d, e, f: None)
    stub("env::__main_argc_argv", [i32]*2, [i32], lambda a, b: 0)
    # Minimal tracedump: the deployed runtime also just hands the buffer
    # off; the callback itself is part of the tracing cost.
    stub("allspark:trace::prog_tracedump", [i32]*2, [], lambda a, b: None)

    instance = linker.instantiate(store, module)
    exports = instance.exports(store)
    state["memory"] = exports["memory"]
    execute = exports["execute"]

    # Init (probe-established): SERVICE event 8, phases 0..2.
    for phase in range(3):
        execute(store, 8, phase, 0)
    ein = next(i for i, a in addr_by_iod.items() if "realtime_data/input" in a)
    return store, execute, regions, ein


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ticks", type=int, default=100_000)
    ap.add_argument("--warmup", type=int, default=2_000)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(DATA / f"{SESSION}.parquet",
                         columns=["current_angle", "current_x"])
    theta = df["current_angle"].values
    x = df["current_x"].values
    n = min(args.n_ticks + args.warmup, len(theta))
    counts = ((theta[:n] * 8192 / (2 * math.pi)).astype(np.int64) & 0x1FFF)
    posiw = (x[:n] * 1e7).astype(np.int64)

    engine = Engine(Config())
    results = {}
    for label, path in (("instrumented", CAPTURE / "module.wasm"),
                        ("original", CAPTURE / "orig_module.wasm")):
        store, execute, regions, ein = build_instance(engine, path)
        region = regions[ein]
        times = np.empty(n, dtype=np.float64)
        for t in range(n):
            struct.pack_into("<h", region, 10, int(counts[t]) - 4096)
            struct.pack_into("<H", region, 12, 0xC000)
            struct.pack_into("<i", region, 14, int(posiw[t]))
            t0 = time.perf_counter_ns()
            execute(store, 0, 0, 0)
            times[t] = time.perf_counter_ns() - t0
        times = times[args.warmup:] / 1000.0   # microseconds
        results[label] = {
            "n": int(len(times)),
            "median_us": float(np.median(times)),
            "p95_us": float(np.quantile(times, .95)),
            "p99_us": float(np.quantile(times, .99)),
            "mean_us": float(times.mean()),
        }
        r = results[label]
        print(f"{label:12s} median {r['median_us']:8.2f} us   "
              f"p99 {r['p99_us']:8.2f} us")

    results["slowdown"] = {
        k: results["instrumented"][f"{k}_us"] / results["original"][f"{k}_us"]
        for k in ("median", "p95", "p99", "mean")}
    print("slowdown:", {k: f"{v:.2f}x" for k, v in results["slowdown"].items()})

    out = Path(__file__).parent / "overhead_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
