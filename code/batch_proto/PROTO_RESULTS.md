# Batch-Evaluation Scheduling Prototype — Results

Date: 2026-09-23 | Code: `batch_proto.cu` | Platform: RTX 3060 Ti (sm_86), CUDA 12.5

## Prototype structure

- **Synthetic data**: N = 8,388,608 transactions × d = 512 items, with 80 planted correlated patterns of length 2–6 (frequency 0.2%–1.5%) + 0.02% noise items; minsup = 0.15%·N = 12,582. The bitmap layout is identical to AnyFIM's.
- **Level-wise search**: at each level, the surviving prefix masks (AND of per-item bitmaps, `mask_build_kernel`) are stacked into matrix A; a single matrix multiply yields the supports of the whole candidate batch against **all** extension items; candidates are pruned by minsup and the next level is expanded in canonical order.
- **Dispatcher** (thresholds from the microbenchmark measurements):
  `b < 64` → bitmap kernel (fallback); tiles < 76 (cannot feed 38 SMs) → BMMA naive;
  `B > 256 MB` → BMMA tiled; otherwise naive.
- **Control**: `--bitmap-only` forces the bitmap kernel throughout.

## Results (both modes produce identical output: 1904 frequent itemsets)

| Level | #prefixes | dispatched kernel | dispatched eval | bitmap-only eval |
|---:|---:|:---:|---:|---:|
| L2 | 248 | BMMA-naive | 26.0 ms | 1078.5 ms |
| L3 | 585 | BMMA-smem | 73.8 ms | 2662.7 ms |
| L4 | 587 | BMMA-smem | 75.0 ms | 2629.9 ms |
| L5 | 350 | BMMA-naive | 40.1 ms | 1538.1 ms |
| L6 | 117 | BMMA-naive | 22.0 ms | 376.6 ms |
| L7 | 17 | **bitmap (fallback triggered)** | 31.5 ms | 31.6 ms |
| **Total** | | | **268.3 ms** | **8317.3 ms** |

**End-to-end counting speedup: 31.0x** (mask construction is identical in both modes, only ~13 ms).

## Correctness (all checks passed)

- The first batch of BMMA results at every level matches the bitmap kernel element by element
- Frequent itemsets satisfy downward closure (the Apriori property)
- All 80 planted patterns recalled (80/80), no false negatives or false positives — fully consistent with exact support counting

## Validated design points

1. **The batch-threshold fallback is sound and cheap**: L7 has only 17 prefixes, not enough to fill a 64-row tile, so it automatically falls back to the bitmap kernel at 31.5 ms — same order as the BMMA path, no cliff.
2. **All three dispatch paths are triggered naturally** — the dispatcher rules are effective, and their parameters come directly from microbenchmark data.
3. One matrix multiply computes the supports of all extension items, eliminating the per-candidate loop; the share of counting in end-to-end time drops from > 99% (bitmap-only) to a controlled range, moving the bottleneck back to search/scheduling logic.
4. Mask construction (AND of k bitmaps per prefix) currently costs only ~5%; at scales beyond ten million transactions it should be re-evaluated (e.g. incremental AND from the parent mask to avoid recomputation per level).

## Reproduce

```
build.bat                       # vcvars64 + nvcc -O3 -arch=sm_86
batch_proto.exe                 # dispatched mode
batch_proto.exe --bitmap-only   # bitmap-only control
```
