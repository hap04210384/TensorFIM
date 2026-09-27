# TensorFIM — Reproducibility Package

This package reproduces the experiments of:

> **TensorFIM: Exact Maximal Frequent Itemset Mining on Tensor Cores** (TKDE submission)

The central claim is verifiable bit-exactly on any NVIDIA GPU from sm_80
upward: tensor-core (b1 BMMA) support counting produces *identical* maximal
frequent itemsets to the classical bitmap engine, and every script in this
package re-checks that identity against archived gold references.

## Environment

- GPU: any NVIDIA GPU with compute capability >= 8.0 (Ampere or newer).
  Reference numbers were measured on an RTX 3060 Ti (sm_86, 38 SMs).
- CUDA toolkit >= 12.0, a C++17 host compiler (MSVC on Windows, GCC on Linux).
- Python >= 3.8 for the driver scripts (standard library only).
- Prebuilt Windows x64 binaries (`code/engine/run/`) are included for
  convenience; they are statically linked and run without a CUDA installation.

## Layout

```
code/
  microbench/   bmma_bench.cu — isolated BMMA vs. bitmap-loop microbenchmark (Table I/II, Fig. 2)
  batch_proto/  batch_proto.cu — wave-batching prototype on synthetic data (Section V-C)
  engine/       kernel_bitmap.cu (baseline) + kernel_bmma.cu (TensorFIM), build scripts
scripts/
  run_all_configs.py       13 dataset-threshold configurations, both engines (Table IV/V, Fig. 3/5)
  run_ablation.py          wave-size sweep + split-k on/off (Table VI)
  run_threshold_sweep.py   accidents/kosarak/connect across thresholds (Fig. 6)
  verify_all.py            set-identity check of all produced results vs. gold
data/
  small/          the seven small-to-mid FIMI benchmarks
  gold_results/   reference MFI sets for every configuration (the correctness anchor)
  make_replica.py, strip_tcga.py, DOWNLOAD.md   large-dataset generation/provenance
```

## Quickstart

```bash
# 1. build the engines (or use the prebuilt binaries in code/engine/run/)
cd code/engine && ./build.bat          # nvcc -O3 -arch=native; MSVC + CUDA on Windows

# 2. microbenchmark: BMMA counting throughput (Table I/II, Fig. 2)
cd ../microbench && ./build.bat && ./bmma_bench.exe

# 3. end-to-end: all 13 configurations, 5 interleaved reps, set-verified
cd ../.. && python scripts/run_all_configs.py

# 4. ablation and threshold sweep
python scripts/run_ablation.py
python scripts/run_threshold_sweep.py

# 5. re-verify every result against the gold references
python scripts/verify_all.py
```

Expected outcome of every correctness gate: `ALL SETS IDENTICAL`.
Timings vary with hardware; the reference medians (RTX 3060 Ti) are below.

## Claim-to-command map

| Paper claim | Command | Reference (RTX 3060 Ti) |
|---|---|---|
| BMMA sustains 56.2 Tbit-op/s, 25.8x geomean over bitmap loop | `code/microbench/bmma_bench.exe` | Table I/II, Fig. 2 |
| Prototype end-to-end 31.0x, identical output | `code/batch_proto/batch_proto.exe` | Section V-C |
| End-to-end speedups 1.4x–12.4x across 13 configurations | `scripts/run_all_configs.py` | Table IV, Fig. 3 |
| Bit-exact output on all configurations | every script + `verify_all.py` | Table III |
| Phase breakdown: BMMA dominates post-compaction | engine logs (`BMMA phase ms:` line) | Fig. 4, Section V-F |
| Amdahl ceiling 25.2x on pumsb_x256 | cost model from phase logs | Section V-F |
| Split-k off = 24.9x slower on pumsb_x256 | `scripts/run_ablation.py` | Table VI |
| Threshold-sensitivity crossover | `scripts/run_threshold_sweep.py` | Fig. 6 |

## Notes

- The engines exit with code 1 after a successful run (historical convention);
  success is the produced `<dataset>-<thr>=Results.txt`, not the exit code.
- pumsb_x256 and webdocs are large (4.3 GB / 1.5 GB); see `data/DOWNLOAD.md`.
  All other configurations run in seconds.
- Baselines: GMiner (https://github.com/opensourcesavvy/GMiner; see paper
  ref [7]) and FPmax*
  (paper ref [18]) are third-party codes and not redistributed here; Table V's
  baseline columns were measured on the same host with the protocols described
  in Section V-A.
