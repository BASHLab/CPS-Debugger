# Reproducing the paper's numbers

Every experimental figure in the paper comes from a macro in
`macros_numbers.tex`, and `gen_numbers.py` writes that file from the result
artifacts under `outputs/`. Nothing in the paper is typed by hand.

```bash
export CPSD_REPO=$(git rev-parse --show-toplevel)      # this checkout
export CPSD_DATA=/path/to/dataset                      # the release, unpacked
export CPSD_CAPTURE=/path/to/raw/captures              # optional, for replay
python3 reproduce/gen_numbers.py --out /tmp/macros_numbers.tex
```

The dataset is a release asset:
https://github.com/BASHLab/CPS-Debugger/releases/tag/dataset-v1

## Which script produces which claim

| Claim in the paper | Script |
|---|---|
| Fault corpus: 408 candidate sites | `outputs/sil_demo/enumerate_mutations.py` |
| 203 reroute on the logged trajectory | `outputs/sil_demo/run_corpus.py` |
| Layout-preserving binary mutation | `outputs/sil_demo/patch_wasm.py` |
| Replay of a mutated binary against logged sensors | `outputs/sil_demo/replay.py` |
| Localization, exact reference (69%) | `outputs/sil_demo/edge_localize.py` |
| Localization, reconstructed reference (46%) | `outputs/sil_demo/edge_localize_recon.py` |
| Frame selection, the 36 to 69 step | `outputs/sil_demo/improve_localize.py` |
| Trivial floors: random 10%, busiest function 36% | `outputs/sil_demo/localize_baselines.py` |
| Alignment costs and the normal substitution profile | `outputs/sil_demo/calibrate_edit_costs.py`, `calibrate_normal_subs.py` |
| Aligning a reconstruction to a replayed trace | `outputs/sil_demo/recon_align.py` |
| Per-position reliability of the reconstruction | `outputs/sil_demo/recon_reliability.py` |
| Closed loop: 400 ran, 247 reroute, 129 silent | `outputs/sil_demo/cosim_faults.py` |
| Whether silent faults are dormant or inert | `outputs/sil_demo/dormancy.py` |
| Repair across five language models | `outputs/sil_demo/repair_study.py`, `llm_patch.py` |
| Divergence removed by a repair | `outputs/sil_demo/repair_residual.py` |
| Plant identification from logged data | `outputs/sil_demo/calibrate_plant.py`, `plant.py` |
| Instrumentation overhead | `outputs/sil_demo/overhead_bench.py` |

## Not included

`cosim.py`, the earlier co-simulation, drives a Bosch Simulink plant through the
MATLAB Engine and is not released. The closed-loop results in the paper do not
depend on it: `cosim_faults.py` runs against the plant identified from the
machine's own logged data (`calibrate_plant.py`, `plant.py`), and imports no
MATLAB.

The operator study is not included. It involves human participants and its
materials are held separately.

## Paths

These scripts previously hardcoded absolute paths on the machine they were
written on. They now read `CPSD_REPO`, `CPSD_DATA` and `CPSD_CAPTURE`, each
falling back to a location relative to this checkout. If a script cannot find an
artifact, set those three and try again.
