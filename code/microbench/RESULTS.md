# BMMA Microbenchmark Results — RTX 3060 Ti (sm_86, CUDA 12.5)

Date: 2026-09-23 | Code: `bmma_bench.cu` | Raw output: `sweep_results.txt` (v1 naive), `sweep_results_v2.txt` (three-way comparison of v1 + v2 + bitmap)

## Verdict

**The approach works — green light.** Both WMMA b1 kernels are clearly faster than the AnyFIM-style hand-written bitmap loop at every shape, with results **bit-exact** against the CPU reference (AND + popcount, no precision loss). The shared-memory tiled variant reaches **53–56 Tbit-op/s** on large shapes, up to **62.7x** over the bitmap loop; picking the better of the two kernels per shape gives a geometric-mean speedup of **25.8x** (7 configurations).

## Workload

C(b×d) = A(b×N bits) × B(N×d bits, stored column-wise as per-item bitmaps); each C element = popcount(AND) of two N-bit vectors. All three kernels compute exactly the same C:

- `bmma_kernel` (v1 naive): WMMA b1 fragments 8×8×128, fragments loaded from global memory with N-bit strides
- `bmma_smem_kernel` (v2 tiled): 64×64 output tiles, cooperative coalesced loads of 64×2048-bit A/B tiles into shared memory, fragments loaded from shared memory
- `bitmap_kernel`: 64-bit AND + `__popcll` loop (same shape as AnyFIM's `getResFrequencyCudaKernel`)

## Measured data (best-of-N; throughput unit: 1 bit-op = one 1-bit AND + accumulate)

| N (transactions) | d (items) | b (batch) | v1 naive (Tbit/s) | v2 tiled (Tbit/s) | bitmap loop (Tbit/s) | best speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 M | 256 | 128 | 12.7 | 8.6 | 3.7 | 3.39x |
| 1.0 M | 256 | 1024 | **55.9** | 42.7 | 1.6 | 34.46x |
| 4.2 M | 256 | 1024 | 46.8 | 41.7 | 1.1 | 43.21x |
| 4.2 M | 1024 | 1024 | 5.9 | **53.0** | 1.0 | **54.94x** |
| 16.8 M | 1024 | 1024 | 5.4 | **53.3** | 0.9 | **62.71x** |
| 67.1 M | 256 | 128 | **12.7** | 8.3 | 1.3 | 10.16x |
| 1.0 M | 16384 | 1024 | 6.0 | **56.2** | 1.3 | **42.66x** |

(16.8M × d4096 × b128 skipped: 8.9 GB exceeds the memory budget.)

## Interpretation

1. **The tiling payoff materializes exactly as predicted**: v1 drops to ~6 Tbit/s when matrix B is large (> 0.5 GB) — b1 fragments load with N-bit strides, i.e. scattered 16-byte accesses; after v2 switches global accesses to coalesced tile loads, those shapes all reach **53–56 Tbit/s**, a 9–10x improvement. The bottleneck is indeed the memory-access pattern, not tensor-core throughput.
2. **The two kernels are complementary and need a dispatcher**: when B is small and cache-resident (d = 256), v1 wins (55.9 vs 42.7); when there are too few tiles to feed 38 SMs (64M × d256 × b128, only 8 tiles), v1 also wins. A production kernel should choose between the two based on "B bytes / tile occupancy" — structurally the same idea as AnyFIM's existing measurement-driven dispatcher.
3. The larger the scale and the batch, the more stable the advantage: 43–63x at tens of millions of transactions with b = 1024, directly supporting the batch-evaluation design.
4. The bitmap-loop baseline sits at 0.9–3.7 Tbit/s across all shapes — same shape and same order of magnitude as AnyFIM's bitmap kernel, so the comparison is fair.
5. Headroom: the theoretical BMMA peak of the 3060 Ti is far above 56 Tbit/s; cp.async double buffering, larger k-chunks, and warp-level multi-row striping could push further. The remaining optimizations serve as ablation material.

## Pitfalls (read before writing the production kernel)

1. **The `ldm` unit of b1 fragments is bits**, not 32-bit words (must be a multiple of 128); this project pads N to a multiple of 4096 bits throughout.
2. **`bmma_sync` defaults to XOR** as the bit operation — you must explicitly pass `wmma::experimental::bmmaBitOpAND` + `bmmaAccumulateOpPOPC`.
3. The b1 load/store API is `load_matrix_sync` / `store_matrix_sync` (not `load_matrix`).
4. Column-major B (one contiguous N-bit segment per item) **maps zero-copy** onto AnyFIM's existing bitmap storage.
5. All-ones data masks layout bugs — correctness validation must use sparse random data with a three-way CPU reference check.
6. The tiled kernel requires b and d to be multiples of 64 (tile alignment); pad otherwise — padded bits are 0 and do not affect counts.
7. **A tiled kernel containing `__syncthreads` must not `return` early per row**: with b not a multiple of 64, the exited warps no longer participate in block synchronization — undefined behavior that silently produces wrong results. All threads must reach every sync point; mask rows only at store time (fixed in the source on 2026-09-23; the batching prototype hit this bug in practice at b = 585).

## Reproduce

```
build.bat            # invokes vcvars64 + nvcc -O3 -arch=sm_86
bmma_bench.exe       # correctness gate + built-in sweep
bmma_bench.exe N d b # single configuration (cross-check)
```
