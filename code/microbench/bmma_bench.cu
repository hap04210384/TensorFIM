// ============================================================================
// BMMA (Tensor Core b1) vs hand-written bitmap loop microbenchmark
// Target: RTX 3060 Ti (sm_86), CUDA 12.5
//
// Workload mirrors the planned AnyFIM mapping:
//   A : b x N 0/1 bit matrix  (b candidate prefix masks, each N bits, row-major)
//   B : d x N 0/1 bit matrix  (d item bitmaps, each N bits contiguous = N x d col-major)
//   C = A AND-multiply B : b x d support counts  (exact popcount, int32)
//
// Two kernels compute the identical C:
//   1. bmma_kernel    : nvcuda::wmma b1 fragments, 8x8x128 tiles (AND + popcount)
//   2. bitmap_kernel  : classic 64-bit AND + __popcll loop (same shape as the
//                       getResFrequencyCudaKernel in AnyFIM kernel_bitmap.cu)
//
// Build:  nvcc -O3 -arch=sm_86 -o bmma_bench.exe bmma_bench.cu
// Run:    bmma_bench.exe                 -> built-in sweep (memory-guarded)
//         bmma_bench.exe N d b           -> single config (bits, items, candidates)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <mma.h>
#include <chrono>

using namespace nvcuda;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

// ---------------------------------------------------------------------------
// Config / layout helpers
// ---------------------------------------------------------------------------
static size_t padBits(size_t n) { return (n + 4095) / 4096 * 4096; }  // N padded to 4096 bits

// ---------------------------------------------------------------------------
// Kernel 1: BMMA via WMMA b1 (8x8x128 fragments).
// Each warp computes an 8-row x 64-col strip of C; block of 8 warps => 64x64 tile.
// A row-major: row i at A + i*W32 (uint32 words). ldm = W32 (must be %4==0).
// B col-major: item j's bitmap at B + j*W32. For col_major K=128 fragment, ldm = W32.
// ---------------------------------------------------------------------------
__global__ void bmma_kernel(const unsigned* __restrict__ A,
                            const unsigned* __restrict__ B,
                            int* __restrict__ C,
                            int b, int d, int W32)
{
    const int warpId = threadIdx.x / 32;
    const int row0 = blockIdx.x * 64 + warpId * 8;   // this warp's 8 rows of A
    const int col0 = blockIdx.y * 64;                // 64 columns of B
    if (row0 >= b) return;

    wmma::fragment<wmma::matrix_a, 8, 8, 128, wmma::experimental::precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, wmma::experimental::precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag[8];

    #pragma unroll
    for (int j = 0; j < 8; j++) wmma::fill_fragment(c_frag[j], 0);

    const unsigned* aBase = A + (size_t)row0 * W32;

    for (int kw = 0; kw < W32; kw += 4) {            // 4 words = 128 bits per k-slice
        wmma::load_matrix_sync(a_frag, aBase + kw, W32 * 32);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const unsigned* bBase = B + (size_t)(col0 + j * 8) * W32 + kw;
            wmma::load_matrix_sync(b_frag, bBase, W32 * 32);
            wmma::bmma_sync(c_frag[j], a_frag, b_frag, c_frag[j],
                            wmma::experimental::bmmaBitOpAND,
                            wmma::experimental::bmmaAccumulateOpPOPC);
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int col = col0 + j * 8;
        if (col < d)
            wmma::store_matrix_sync(C + (size_t)row0 * d + col, c_frag[j], d, wmma::mem_row_major);
    }
}

// ---------------------------------------------------------------------------
// Kernel 1b: BMMA with shared-memory tiling.
// Block computes a 64x64 C tile (8 warps, warp = 8x64 strip, same as naive).
// Each iteration stages a 64-row x 2048-bit chunk of A and a 64-item x 2048-bit
// chunk of B into shared memory with fully coalesced global reads, then all
// fragment loads hit shared memory (ldm = 2048 bits). Requires b,d % 64 == 0.
// ---------------------------------------------------------------------------
#define TILE_K_WORDS 64                       // 2048 bits per k-chunk
#define TILE_K_BITS  (TILE_K_WORDS * 32)

__global__ void bmma_smem_kernel(const unsigned* __restrict__ A,
                                 const unsigned* __restrict__ B,
                                 int* __restrict__ C,
                                 int b, int d, int W32)
{
    __shared__ unsigned As[64][TILE_K_WORDS];
    __shared__ unsigned Bs[64][TILE_K_WORDS];

    const int warpId = threadIdx.x / 32;
    const int tid    = threadIdx.x;
    const int row0   = blockIdx.x * 64 + warpId * 8;
    const int col0   = blockIdx.y * 64;
    const bool active = (row0 < b);   // inactive warps must still reach __syncthreads

    wmma::fragment<wmma::matrix_a, 8, 8, 128, wmma::experimental::precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, wmma::experimental::precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) wmma::fill_fragment(c_frag[j], 0);

    const unsigned* gA = A + (size_t)(blockIdx.x * 64) * W32;
    const unsigned* gB = B + (size_t)(blockIdx.y * 64) * W32;

    for (int kw = 0; kw < W32; kw += TILE_K_WORDS) {
        // coalesced staging: 64 rows x 64 words = 4096 words each, 256 threads
        #pragma unroll
        for (int idx = tid; idx < 64 * TILE_K_WORDS; idx += 256) {
            const int r = idx / TILE_K_WORDS, c = idx % TILE_K_WORDS;
            As[r][c] = __ldg(gA + (size_t)r * W32 + kw + c);
            Bs[r][c] = __ldg(gB + (size_t)r * W32 + kw + c);
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K_WORDS; k += 4) {     // 128-bit k-slices, in smem
            wmma::load_matrix_sync(a_frag, &As[warpId * 8][k], TILE_K_BITS);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                wmma::load_matrix_sync(b_frag, &Bs[j * 8][k], TILE_K_BITS);
                wmma::bmma_sync(c_frag[j], a_frag, b_frag, c_frag[j],
                                wmma::experimental::bmmaBitOpAND,
                                wmma::experimental::bmmaAccumulateOpPOPC);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int col = col0 + j * 8;
        if (active && col < d)
            wmma::store_matrix_sync(C + (size_t)row0 * d + col, c_frag[j], d, wmma::mem_row_major);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: hand-written bitmap loop (same pattern as AnyFIM's GPU kernel).
// One block per candidate row i, threads over items j; each thread AND+popcounts
// its way down the two N-bit vectors.
// ---------------------------------------------------------------------------
__global__ void bitmap_kernel(const unsigned long long* __restrict__ A,
                              const unsigned long long* __restrict__ B,
                              int* __restrict__ C,
                              int b, int d, int W64)
{
    const int i = blockIdx.x;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= d) return;

    const unsigned long long* a = A + (size_t)i * W64;
    const unsigned long long* v = B + (size_t)j * W64;
    unsigned long long s = 0;
    for (int w = 0; w < W64; w++)
        s += __popcll(__ldg(a + w) & __ldg(v + w));
    C[(size_t)i * d + j] = (int)s;
}


// ---------------------------------------------------------------------------
// CPU reference (exact, slow; used only on the small validation config)
// ---------------------------------------------------------------------------
static void cpu_ref(const unsigned* A, const unsigned* B, int* C,
                    int b, int d, int W32)
{
    for (int i = 0; i < b; i++)
        for (int j = 0; j < d; j++) {
            const unsigned* a = A + (size_t)i * W32;
            const unsigned* v = B + (size_t)j * W32;
            unsigned s = 0;
            for (int w = 0; w < W32; w++) s += __popcnt(a[w] & v[w]);
            C[(size_t)i * d + j] = (int)s;
        }
}

// ---------------------------------------------------------------------------
// Host utilities
// ---------------------------------------------------------------------------
static void fill_random_bits(unsigned* p, size_t words, unsigned densityPct, unsigned long long& seed)
{
    (void)densityPct;
    // two independent xorshift streams ANDed => ~25% bit density, O(1) per word
    unsigned long long s2 = seed * 0x2545F4914F6CDD1DULL + 0x123456789ULL;
    for (size_t w = 0; w < words; w++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        s2   ^= s2 << 13;   s2   ^= s2 >> 7;   s2   ^= s2 << 17;
        p[w] = (unsigned)(seed & s2);
        p[w] &= (unsigned)(seed >> 32);   // fold upper half => density ~12.5%
    }
}

// ---------------------------------------------------------------------------
// Sustained-load throttle probe: run the smem BMMA kernel back-to-back for a
// fixed wall-clock budget and log per-iteration timing as CSV. An export-
// limiter throttle (e.g. RTX 5090D AI lock) shows up as a step increase in
// per-iteration time after ~3 s of sustained tensor-core load.
//   bmma_bench.exe sustain N d b seconds
// ---------------------------------------------------------------------------
static int sustain_mode(size_t Nraw, int d, int b, double seconds)
{
    size_t N = padBits(Nraw);
    int    W32 = (int)(N / 32);
    size_t bytesA = (size_t)b * W32 * 4, bytesB = (size_t)d * W32 * 4, bytesC = (size_t)b * d * 4;
    printf("sustain probe: N=%zu bits  d=%d  b=%d  budget=%.0f s  A=%.2f GB  B=%.2f GB\n",
           N, d, b, seconds, bytesA / 1e9, bytesB / 1e9);

    std::vector<unsigned> hA((size_t)b * W32), hB((size_t)d * W32);
    unsigned long long seed = 0x9E3779B97F4A7C15ULL;
    fill_random_bits(hA.data(), hA.size(), 30, seed);
    fill_random_bits(hB.data(), hB.size(), 30, seed);

    unsigned *dA, *dB; int* dC;
    CK(cudaMalloc(&dA, bytesA)); CK(cudaMalloc(&dB, bytesB)); CK(cudaMalloc(&dC, bytesC));
    CK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    dim3 grid((b + 63) / 64, (d + 63) / 64);
    // one-shot correctness cross-check before the sustained loop
    bmma_smem_kernel<<<grid, 256>>>(dA, dB, dC, b, d, W32);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    {
        std::vector<int> hC1((size_t)b * d, -1), hC2((size_t)b * d, -1);
        CK(cudaMemcpy(hC1.data(), dC, bytesC, cudaMemcpyDeviceToHost));
        bitmap_kernel<<<dim3(b, (d + 255) / 256), 256>>>((const unsigned long long*)dA,
            (const unsigned long long*)dB, dC, b, d, (int)(N / 64));
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hC2.data(), dC, bytesC, cudaMemcpyDeviceToHost));
        if (hC1 != hC2) { fprintf(stderr, "  [FAIL] smem vs bitmap mismatch, aborting sustain\n"); return 1; }
        printf("  [OK] smem == bitmap cross-check\n");
    }

    const double bitOps = (double)b * d * N;
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    // warmup launch, then start the wall clock
    bmma_smem_kernel<<<grid, 256>>>(dA, dB, dC, b, d, W32);
    CK(cudaDeviceSynchronize());
    const std::chrono::steady_clock::time_point wall0 = std::chrono::steady_clock::now();

    printf("iter,wall_s,kernel_ms,tbit_ops\n");
    std::vector<double> walls, mss;
    for (long it = 0; ; it++) {
        double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
        if (wall >= seconds) break;
        CK(cudaEventRecord(t0));
        bmma_smem_kernel<<<grid, 256>>>(dA, dB, dC, b, d, W32);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
        walls.push_back(wall); mss.push_back(ms);
        printf("%ld,%.3f,%.3f,%.2f\n", it, wall, (double)ms, bitOps / (ms * 1e-3) / 1e12);
        if (it % 10 == 0) fflush(stdout);
    }
    fflush(stdout);

    // verdict: mean kernel time in the first 3 s vs after 3 s
    double eSum = 0, lSum = 0; int eN = 0, lN = 0;
    for (size_t i = 0; i < mss.size(); i++) {
        if (walls[i] <= 3.0) { eSum += mss[i]; eN++; } else { lSum += mss[i]; lN++; }
    }
    if (eN && lN) {
        double eMs = eSum / eN, lMs = lSum / lN, ratio = lMs / eMs;
        printf("verdict: first3s=%.1f ms (n=%d, %.1f Tbit-op/s)  after3s=%.1f ms (n=%d, %.1f Tbit-op/s)  ratio=%.3f -> %s\n",
               eMs, eN, bitOps / (eMs * 1e-3) / 1e12,
               lMs, lN, bitOps / (lMs * 1e-3) / 1e12,
               ratio, ratio > 1.15 ? "THROTTLE SUSPECTED" : "no throttling detected");
    } else {
        printf("verdict: not enough iterations in one of the windows (n_early=%d, n_late=%d)\n", eN, lN);
    }
    CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
struct BenchResult {
    double bmmaMs, smemMs, bitmapMs;
    bool   ok;
};

static BenchResult run_config(size_t Nraw, int d, int b, bool check)
{
    BenchResult res = {0, 0, 0, true};
    size_t N   = padBits(Nraw);
    int    W32 = (int)(N / 32);
    int    W64 = (int)(N / 64);
    size_t bytesA = (size_t)b * W32 * 4;
    size_t bytesB = (size_t)d * W32 * 4;
    size_t bytesC = (size_t)b * d * 4;

    printf("--------------------------------------------------------------------------\n");
    printf("N=%zu bits (%.1f M tx)  d=%d items  b=%d candidates  |  A=%.2f GB  B=%.2f GB\n",
           N, N / 1e6, d, b, bytesA / 1e9, bytesB / 1e9);

    std::vector<unsigned> hA((size_t)b * W32), hB((size_t)d * W32);
    unsigned long long seed = 0x9E3779B97F4A7C15ULL;
    fill_random_bits(hA.data(), hA.size(), 30, seed);
    fill_random_bits(hB.data(), hB.size(), 30, seed);

    unsigned *dA, *dB; int* dC;
    CK(cudaMalloc(&dA, bytesA));
    CK(cudaMalloc(&dB, bytesB));
    CK(cudaMalloc(&dC, bytesC));
    CK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    std::vector<int> hC_bmma((size_t)b * d, -1), hC_bitmap((size_t)b * d, -1);

    dim3 gridBmma((b + 63) / 64, (d + 63) / 64);
    dim3 gridBitmap(b, (d + 255) / 256);

    // ---- correctness -------------------------------------------------------
    bmma_kernel<<<gridBmma, 256>>>(dA, dB, dC, b, d, W32);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hC_bmma.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    bitmap_kernel<<<gridBitmap, 256>>>((const unsigned long long*)dA,
                                       (const unsigned long long*)dB, dC, b, d, W64);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hC_bitmap.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    if (hC_bmma != hC_bitmap) {
        printf("  [FAIL] BMMA vs bitmap kernel mismatch!\n");
        res.ok = false;
    }
    bmma_smem_kernel<<<gridBmma, 256>>>(dA, dB, dC, b, d, W32);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    {
        std::vector<int> hC_smem((size_t)b * d, -1);
        CK(cudaMemcpy(hC_smem.data(), dC, bytesC, cudaMemcpyDeviceToHost));
        if (hC_smem != hC_bitmap) {
            printf("  [FAIL] BMMA-smem vs bitmap kernel mismatch!\n");
            res.ok = false;
        }
    }
    if (check) {
        std::vector<int> hC_ref((size_t)b * d, -1);
        cpu_ref(hA.data(), hB.data(), hC_ref.data(), b, d, W32);
        if (hC_ref != hC_bmma)   { printf("  [FAIL] BMMA vs CPU reference mismatch!\n");   res.ok = false; }
        if (hC_ref != hC_bitmap) { printf("  [FAIL] bitmap vs CPU reference mismatch!\n"); res.ok = false; }
        if (res.ok) printf("  [OK] CPU reference == BMMA == BMMA-smem == bitmap kernel\n");
    } else if (res.ok) {
        printf("  [OK] BMMA == BMMA-smem == bitmap kernel (cross-check)\n");
    }

    // ---- timing ------------------------------------------------------------
    // adaptive reps: big configs (slow bitmap baseline) get fewer repetitions
    double bitOps0 = (double)b * d * N;
    const int warmup = (bitOps0 > 2e12) ? 1 : 2;
    const int rep    = (bitOps0 > 2e12) ? 2 : 5;
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

    float best;
    best = 1e30f;
    for (int r = 0; r < warmup + rep; r++) {
        CK(cudaEventRecord(t0));
        bmma_kernel<<<gridBmma, 256>>>(dA, dB, dC, b, d, W32);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        if (r >= warmup) best = std::min(best, ms);
    }
    res.bmmaMs = best;

    best = 1e30f;
    for (int r = 0; r < warmup + rep; r++) {
        CK(cudaEventRecord(t0));
        bmma_smem_kernel<<<gridBmma, 256>>>(dA, dB, dC, b, d, W32);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        if (r >= warmup) best = std::min(best, ms);
    }
    res.smemMs = best;

    best = 1e30f;
    for (int r = 0; r < warmup + rep; r++) {
        CK(cudaEventRecord(t0));
        bitmap_kernel<<<gridBitmap, 256>>>((const unsigned long long*)dA,
                                           (const unsigned long long*)dB, dC, b, d, W64);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        if (r >= warmup) best = std::min(best, ms);
    }
    res.bitmapMs = best;
    CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));

    double bitOps = (double)b * d * N;   // 1-bit AND + popcount accumulate
    printf("  BMMA naive : %9.3f ms   %8.1f Tbit-op/s\n", res.bmmaMs,  bitOps / (res.bmmaMs  * 1e-3) / 1e12);
    printf("  BMMA smem  : %9.3f ms   %8.1f Tbit-op/s\n", res.smemMs,  bitOps / (res.smemMs  * 1e-3) / 1e12);
    printf("  bitmap     : %9.3f ms   %8.1f Tbit-op/s\n", res.bitmapMs, bitOps / (res.bitmapMs * 1e-3) / 1e12);
    printf("  speedup: naive %.2fx | smem %.2fx (vs bitmap)\n",
           res.bitmapMs / res.bmmaMs, res.bitmapMs / res.smemMs);
    fflush(stdout);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return res;
}

int main(int argc, char** argv)
{
    int dev = 0; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s  sm_%d%d  SMs=%d  mem=%.1f GB\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.totalGlobalMem / 1e9);
    if (prop.major < 8) { fprintf(stderr, "BMMA b1 needs sm_80+\n"); return 1; }
    const double memBudget = prop.totalGlobalMem * 0.8;

    if (argc == 6 && strcmp(argv[1], "sustain") == 0) {
        return sustain_mode(strtoull(argv[2], nullptr, 10), atoi(argv[3]), atoi(argv[4]), atof(argv[5]));
    }
    if (argc == 4) {
        size_t N = strtoull(argv[1], nullptr, 10);
        int d = atoi(argv[2]), b = atoi(argv[3]);
        run_config(N, d, b, false);   // single-config mode: cross-check only (CPU ref is O(b*d*N))
        return 0;
    }

    // Correctness gate: small config, full CPU cross-check
    printf("=== correctness check (small config, CPU reference) ===\n");
    BenchResult v = run_config(1u << 20, 256, 128, true);
    if (!v.ok) { fprintf(stderr, "Correctness failed, aborting sweep.\n"); return 1; }

    // Sweep: large-scale shapes from the research memo (tens of millions of tx)
    printf("\n=== throughput sweep ===\n");
    struct Cfg { size_t N; int d, b; };
    Cfg cfgs[] = {
        { 1u << 20,   256,  128 },   //  1M tx baseline
        { 1u << 20,   256, 1024 },
        { 1u << 22,   256, 1024 },   //  4M tx
        { 1u << 22,  1024, 1024 },
        { 1u << 24,  1024, 1024 },   // 16M tx, B = 2 GB
        { 1u << 24,  4096,  128 },
        { 1u << 26,   256,  128 },   // 64M tx
        { 1u << 20, 16384, 1024 },   // wide-d (TCGA-like)
    };
    double geo = 1.0; int cnt = 0;
    printf("\n%-8s %-6s %-6s | %9s %9s %9s | %8s\n", "N(Mtx)", "d", "b",
           "naive(Tb)", "smem(Tb)", "bmap(Tb)", "speedup");
    for (Cfg c : cfgs) {
        size_t N = padBits(c.N);
        double bytes = (double)(c.b + c.d) * (N / 32) * 4 + (double)c.b * c.d * 4;
        if (bytes > memBudget) { printf("%-8.1f %-6d %-6d | skipped (%.1f GB > budget)\n",
                                        c.N / 1e6, c.d, c.b, bytes / 1e9); continue; }
        BenchResult r = run_config(c.N, c.d, c.b, false);
        double bitOps = (double)c.b * c.d * N;
        double tn = bitOps / (r.bmmaMs * 1e-3) / 1e12;
        double ts = bitOps / (r.smemMs * 1e-3) / 1e12;
        double tl = bitOps / (r.bitmapMs * 1e-3) / 1e12;
        double sp = r.bitmapMs / std::min(r.smemMs, r.bmmaMs);
        printf("%-8.1f %-6d %-6d | %9.1f %9.1f %9.1f | %7.2fx\n",
               c.N / 1e6, c.d, c.b, tn, ts, tl, sp);
        geo *= sp; cnt++;
    }
    if (cnt) printf("\ngeomean speedup (best BMMA / bitmap loop): %.2fx over %d configs\n",
                    pow(geo, 1.0 / cnt), cnt);
    return 0;
}
