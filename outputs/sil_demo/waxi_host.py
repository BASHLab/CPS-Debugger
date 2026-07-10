"""waxi_host.py — reusable SIL host for the shipped ctrlX controller binary.

Refactored from the proven C0 probe (probe_wasmtime.py, GO on 2026-07-02:
1000 ticks, 99.89% in-vocab). Wraps the WebAssembly controller under
wasmtime with host-side stubs for the 9 waxi:* imports + WASI + the
inline trace-dump callback, and exposes a clean per-tick interface:

    host = WaxiHost(module_path, vocab_path)
    host.service_init()                     # SERVICE phase, opens datalayer
    host.write_input(sensor_bytes)          # 64-byte EtherCAT input image
    triples = host.tick()                   # execute(TICK); decoded trace edges
    out = host.read_output()                # 64-byte EtherCAT output image

The host is deliberately dumb about physics: it moves raw bytes in and out
of the EtherCAT input/output regions and decodes the trace ring buffer.
plant.py owns the sensor synthesis and actuator interpretation.
"""
import json
import struct
from collections import Counter
from pathlib import Path

from wasmtime import (
    Config, Engine, Func, FuncType, Linker, Module, Store, ValType,
    WasiConfig,
)

RING_COUNTER = 131072
RING_RECORDS = 131080

# Datalayer region sizes (bytes), keyed by a substring of the iod address.
REGION_SIZES = {
    "realtime_data/input": 64,
    "realtime_data/output": 64,
    "pendulum-logging/output": 1 << 20,
}
_DEFAULT_REGION = 4096


def _region_size(addr: str) -> int:
    for key, size in REGION_SIZES.items():
        if key in addr:
            return size
    return _DEFAULT_REGION


def decode_triple(rec: int):
    """Decode one 8-byte trace record: i64 (func_id<<48)|(from_pc<<24)|to_pc."""
    return ((rec >> 48) & 0xFFFF, (rec >> 24) & 0xFFFFFF, rec & 0xFFFFFF)


class _Datalayer:
    """Host model of the waxi datalayer: byte regions keyed by an iod handle."""

    def __init__(self):
        self.regions = {}       # iod -> bytearray
        self.addr_by_iod = {}   # iod -> address string
        self.next_iod = 100
        self.calls = Counter()

    def open_memory(self, addr: str) -> int:
        iod = self.next_iod
        self.next_iod += 1
        self.regions[iod] = bytearray(_region_size(addr))
        self.addr_by_iod[iod] = addr
        return iod

    def iod_for(self, addr_substr: str):
        for iod, addr in self.addr_by_iod.items():
            if addr_substr in addr:
                return iod
        return None


def _read_cstr(buf: bytes, ptr: int = 0, maxlen: int = 256) -> str:
    end = buf.find(b"\x00", ptr, ptr + maxlen)
    if end < 0:
        end = min(ptr + maxlen, len(buf))
    return buf[ptr:end].decode("utf-8", "replace")


class WaxiHost:
    """SIL host wrapping the shipped controller wasm under wasmtime."""

    # execute(event, phase, param): TICK=0, SERVICE=8 (from the wat br_table).
    EVENT_TICK = 0
    EVENT_SERVICE = 8

    def __init__(self, module_path, vocab_path=None, quiet=True):
        self.quiet = quiet
        self._engine = Engine(Config())
        self._module = Module.from_file(self._engine, str(module_path))
        self._store = Store(self._engine)

        wasi = WasiConfig()
        if not quiet:
            wasi.inherit_stdout()
            wasi.inherit_stderr()
        self._store.set_wasi(wasi)

        self._linker = Linker(self._engine)
        self._linker.define_wasi()

        self.dlr = _Datalayer()
        self._memory = None
        # Per-dump raw byte chunks captured since the last clear().
        self._dump_chunks = []

        self._define_stubs()

        instance = self._linker.instantiate(self._store, self._module)
        exports = instance.exports(self._store)
        self._memory = exports["memory"]
        self._execute = exports["execute"]

        self.vocab = None
        if vocab_path is not None:
            self.vocab = {tuple(t)
                          for t in json.loads(Path(vocab_path).read_text())["triples"]}

    # ── raw memory helpers ──────────────────────────────────────────────────
    def _mem_read(self, off, n):
        return bytes(self._memory.read(self._store, off, off + n))

    def _mem_write(self, off, data: bytes):
        self._memory.write(self._store, data, off)

    # ── waxi + wasi stubs ────────────────────────────────────────────────────
    def _define_stubs(self):
        store = self._store
        i32 = ValType.i32()

        def stub(name, params, results, fn):
            ft = FuncType(params, results)
            f = Func(store, ft, fn)
            mod_name, func_name = name.split("::")
            self._linker.define(store, mod_name, func_name, f)

        dlr = self.dlr

        def svc_dlr_factory():
            dlr.calls["factory"] += 1
            return 1

        def factory_open_memory(factory, iod_ptr, iod_size, addr_ptr):
            addr = _read_cstr(self._mem_read(addr_ptr, 256))
            iod = dlr.open_memory(addr)
            self._mem_write(iod_ptr, struct.pack("<i", iod))
            return 0

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
            self._mem_write(buf_ptr, bytes(region[off:off + n]))
            dlr.calls["read"] += 1
            return 0

        def user_write_bytes(iod, off, buf_ptr, n):
            region = dlr.regions.get(iod)
            if region is None:
                return 1
            region[off:off + n] = self._mem_read(buf_ptr, n)
            dlr.calls["write"] += 1
            return 0

        def logging_log(a, b, c):
            dlr.calls["log"] += 1

        def logging_log_extended(a, b, c, d, e, f):
            dlr.calls["log_ext"] += 1

        def main_argc_argv(a, b):
            return 0

        def prog_tracedump(a, b):
            dlr.calls["tracedump"] += 1
            if self._memory is not None and b > 0:
                self._dump_chunks.append(self._mem_read(a, b))

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

    # ── lifecycle ────────────────────────────────────────────────────────────
    def service_init(self):
        """Run the SERVICE phases until the datalayer regions are open.

        Mirrors the probe: sweep execute(8, phase, 0); init completes at
        (8, 2, 0) on the shipped binary. Returns the (event, phase) pair.
        """
        for event in (self.EVENT_SERVICE, 7, 6):
            for phase in range(4):
                self._execute(self._store, event, phase, 0)
                if self.dlr.iod_for("ethercat") is not None:
                    self._dump_chunks.clear()   # drop init-phase trace dumps
                    return (event, phase)
        raise RuntimeError("SERVICE init never opened the datalayer memories")

    # ── per-tick I/O ─────────────────────────────────────────────────────────
    def input_region(self):
        return self.dlr.regions[self.dlr.iod_for("realtime_data/input")]

    def output_region(self):
        return self.dlr.regions[self.dlr.iod_for("realtime_data/output")]

    def write_input(self, data: bytes, off: int = 0):
        self.input_region()[off:off + len(data)] = data

    def read_output(self, n: int = 64, off: int = 0) -> bytes:
        return bytes(self.output_region()[off:off + n])

    def tick_raw(self):
        """Run one control tick; return the concatenated raw trace-record
        bytes (8 bytes per control-flow edge). Callers vectorize the decode
        with numpy, which is far faster than per-record struct.unpack."""
        self._dump_chunks.clear()
        self._execute(self._store, self.EVENT_TICK, 0, 0)
        return b"".join(self._dump_chunks)

    def tick(self, decode=True):
        """Run one control tick; return decoded trace triples for this tick.

        With decode=False returns the raw record count only (cheaper).
        """
        self._dump_chunks.clear()
        self._execute(self._store, self.EVENT_TICK, 0, 0)
        triples = []
        n = 0
        for raw in self._dump_chunks:
            for o in range(0, len(raw) - 7, 8):
                (rec,) = struct.unpack("<q", raw[o:o + 8])
                n += 1
                if decode:
                    triples.append(decode_triple(rec))
        return triples if decode else n
