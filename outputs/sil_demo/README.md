# sil_demo — controller replay + fault-injection harness

Runs the shipped WebAssembly controller (`module.wasm`) off-device under
`wasmtime` to reconstruct control-flow traces and study seeded software faults.

## Approach: open-loop trace-differential replay
Rather than simulate the plant in closed loop (which requires faithfully
matching the Bosch physical model, including its stateful stick-slip friction —
see `../../docs_archive/iaai27_work/c2_findings.md`), we **replay a real
session's logged sensors** through the nominal and mutated binaries and compare
the reconstructed traces. No plant model, no sim-to-real gap; detection and
localization run on real operating data.

The controller `module.wasm` runs on the **wasmtime** WebAssembly runtime
through `waxi_host.py`, which supplies Python host functions for the 9 `waxi:*`
datalayer/logging imports and WASI. `replay.py` steps this host one control
tick at a time: it writes the re-encoded sensor bytes into the controller's
EtherCAT input region, calls the binary's `execute(TICK)` export, and reads
back the control-flow-edge records that the in-binary instrumentation appended
to its linear-memory ring buffer, decoding them to `(func_id, src_pc, tgt_pc)`
triples. The nominal and mutated binaries run identically on the same sensor
sequence, so any trace difference comes from the mutation alone.

## Files
- `waxi_host.py` — reusable host: instantiates the wasm controller with the 9
  `waxi:*` + WASI stubs, runs SERVICE init, and returns the decoded per-tick
  control-flow trace (ring-buffer records → `(func_id, src_pc, tgt_pc)` triples).
- `patch_wasm.py` — layout-preserving binary mutations for the fault corpus:
  constant edits (same-width sLEB128/immediate) and single-byte opcode swaps
  (relational / arithmetic). Each verifies the old bytes before writing.
- `replay.py` — encodes real logged `(theta, x)` into the EtherCAT input image
  and replays it through nominal vs mutant binaries; reports trace divergence
  (detection) and the divergent `func_id`s (localization).
- `plant.py`, `run_sil.py`, `calibrate_plant.py` — closed-loop cart-pole SIL
  (retained; parked). The controller reaches vertical but does not hold balance
  without the full Bosch stick-slip friction model. See `c2_findings.md`.
- `probe_wasmtime.py` — C0 go/no-go probe (shipped wasm decodes to in-vocab
  tokens). `overhead_bench.py` — F2' tracing-overhead measurement.

## Validated (2026-07)
`replay.py` with the `balanced_counter` mutation (1500→150, seeded by
`patch_wasm.py`): replaying a real session across the swing-up→balance
transition, the mutant trace diverges from nominal at the tick the shortened
counter trips (~150 ticks into balance) and the divergence localizes to
`func_id 20` (`stays_balanced`), the function that reads the mutated constant.

## Run
```
PY=/home/simran/.conda/envs/slimllm/bin/python   # wasmtime + pyarrow; no torch
$PY patch_wasm.py <module.wasm> /tmp/mutant.wasm --bug balanced_counter
$PY replay.py --mutant /tmp/mutant.wasm --ticks 9000 --start 0
```
