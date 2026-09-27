# TensorFIM USB One-Click Validation Package (v2.1, prebuilt / no-install)

> Changelog: v2.1 (2026-09-26) — both engines carry the ID-compaction fix (supports wide sparse item namespaces such as webdocs; bitmap memory is allocated by the number of surviving items). The sources here are byte-identical to `code/engine/src`. Self-test: accidents_x4 @ 0.40, both engines PASS (762 itemsets, exact match).

## One-line instructions for the volunteer

> Plug in the USB drive → double-click `RUN_ME.bat` → leave the computer alone until it finishes → unplug the drive when you see "DONE". **No software installation is required.**
>
> If anything is wrong with the environment, the script reports the error and stops within the first minute — it never silently wastes your time. If you are short on time, run `RUN_ME.bat quick` in a cmd window (about 30 minutes); the validation value is the same.

## Detailed procedure

1. **Double-click `RUN_ME.bat`** (full mode: 9 configurations, about 1.5–2 hours). If time is tight, run `RUN_ME.bat quick` in a cmd window (quick mode: 3 configurations, about 30 minutes).

2. **No CUDA toolkit or Visual Studio installation is needed**: the package ships precompiled executables (statically linked, covering RTX 30/40/50 series, A100/H100/B-series data-center GPUs, with a PTX fallback). As long as the NVIDIA GPU works normally on the machine (driver installed), the benchmarks run.
   - In the rare case that a prebuilt binary cannot start (e.g. a very old driver), the package falls back to on-site compilation; only then does it need network access to install a build toolchain (follow the on-screen instructions and approve the UAC prompt).

3. **Do not use the computer while it runs**: no browser, no games — preferably lock the screen and leave it. Timing data is sensitive to interference.

4. Unplug the drive after you see `DONE. It is now safe to remove the USB drive.`

## What the batch script does automatically

1. Detects the GPU model and architecture (cards that are too old automatically run the baseline engine only, with a note), and saves a full GPU snapshot (driver, clocks, power cap).
2. For each dataset configuration: runs both engines 3 times each (including the sparse real-world dataset kosarak and the biomedical dataset TCGA_BRCA).
3. **Sustained-load probe** (30 s): records per-second tensor-core throughput to check whether the "AI compute cap" of China-market cards such as the RTX 5090D affects this workload.
4. **Peak-throughput sweep** (a few minutes): BMMA microbenchmark at 8 sizes, measuring the absolute tensor-core performance of the card (used for the cross-platform table in the paper).
5. Compares the mining results against the archived references **itemset by itemset** (exact match required; a mismatch is explicitly reported as FAIL — the script never passes with bad results).
6. Generates `REPORT.txt` (machine info + per-configuration PASS/FAIL + median runtime).

## What to bring back

- The `runs\` folder — **each computer gets its own subfolder** (named with machine name + GPU model + timestamp), never overwritten; inside are `REPORT.txt` (correctness verdict and timings for that machine), all run logs, and `sustain_probe.csv` (per-second throughput curve).
- The same USB drive can be run on multiple machines in sequence; results are stored separately.

## Notes

- A USB 3.0 (or faster) port is recommended (datasets total about 7 GB).
- Exit code 1 from the engine executable is pre-existing behavior and does not affect the results.
- The package is GPU-agnostic: any NVIDIA card (GTX 10 series through RTX 50 series) works; RTX 20 series and older run the baseline engine only.
- If `REPORT.txt` shows `MISSING` or `FAIL`, simply bring the drive back — the data itself has diagnostic value.
