// ============================================================================
// Batch candidate evaluation prototype for BMMA-accelerated FIM
// Target: RTX 3060 Ti (sm_86), CUDA 12.5
//
// Pipeline (memo section 4 mapping):
//   synthetic bitmap dataset (80 planted correlated patterns, N=8.4M tx)
//   -> level-wise search: survivor prefix masks stacked into A (b x N bits)
//   -> dispatcher picks kernel per batch:
//        b < 64           -> bitmap loop kernel  (fallback, memo risk #1)
//        grid blocks < 76 -> BMMA v1 naive        (SMs underfed otherwise)
//        B bytes > 256 MB -> BMMA v2 smem-tiled
//        else             -> BMMA v1 naive
//   -> one matrix product yields supports of ALL extensions of the whole batch
//   -> prune by minsup, canonical extension, next level
//
// Validation: per-level cross-check BMMA vs bitmap kernel; downward-closure
// check; planted-pattern recall check against exact host-computed supports.
// End-to-end: dispatch mode vs bitmap-only mode wall time.
//
// Build: nvcc -O3 -arch=sm_86 -std=c++17 -o batch_proto.exe batch_proto.cu
// Run:   batch_proto.exe            -> dispatch mode + bitmap-only comparison
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <set>
#include <algorithm>
#include <windows.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while (0)

// ------------------------------- parameters ---------------------------------
static const int    D_ITEMS   = 512;              // total items (multiple of 64)
static const size_t N_TX      = 1u << 23;         // 8.4M transactions
static const int    N_PAT     = 80;               // planted patterns
static const int    MAXPAT    = 6;                // max pattern size
static const int    MAXK      = 8;                // search deeper than patterns to exercise fallback
static const int    BATCH     = 1024;             // max prefixes per batch
static const double MINSUP_PC = 0.15;             // minsup in percent of N
static const int    FALLBACK_B = 64;              // below this batch -> bitmap kernel

// ------------------------------- GPU kernels --------------------------------

// count support of every single item (for F1)
__global__ void item_count_kernel(const unsigned* __restrict__ B, int d, int W32, int* cnt)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d) return;
    const unsigned* v = B + (size_t)j * W32;
    unsigned s = 0;
    for (int w = 0; w < W32; w++) s += __popc(__ldg(v + w));
    cnt[j] = (int)s;
}

// A[i] = AND of the k item bitmaps listed in items[i*k .. i*k+k-1]; one block per row
__global__ void mask_build_kernel(const unsigned* __restrict__ B,
                                  const int* __restrict__ items, int k,
                                  int W32, unsigned* __restrict__ A)
{
    const int i = blockIdx.x;
    unsigned* a = A + (size_t)i * W32;
    const int* list = items + (size_t)i * k;
    for (int w = threadIdx.x; w < W32; w += blockDim.x) {
        unsigned x = ~0u;
        for (int t = 0; t < k; t++) x &= __ldg(B + (size_t)list[t] * W32 + w);
        a[w] = x;
    }
}

// BMMA v1 naive (fragments straight from global, ldm = N bits)
__global__ void bmma_kernel(const unsigned* __restrict__ A,
                            const unsigned* __restrict__ B,
                            int* __restrict__ C, int b, int d, int W32)
{
    const int warpId = threadIdx.x / 32;
    const int row0 = blockIdx.x * 64 + warpId * 8;
    const int col0 = blockIdx.y * 64;
    if (row0 >= b) return;
    wmma::fragment<wmma::matrix_a, 8, 8, 128, wmma::experimental::precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, wmma::experimental::precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) wmma::fill_fragment(c_frag[j], 0);
    const unsigned* aBase = A + (size_t)row0 * W32;
    for (int kw = 0; kw < W32; kw += 4) {
        wmma::load_matrix_sync(a_frag, aBase + kw, W32 * 32);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            wmma::load_matrix_sync(b_frag, B + (size_t)(col0 + j * 8) * W32 + kw, W32 * 32);
            wmma::bmma_sync(c_frag[j], a_frag, b_frag, c_frag[j],
                            wmma::experimental::bmmaBitOpAND,
                            wmma::experimental::bmmaAccumulateOpPOPC);
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; j++)
        if (col0 + j * 8 < d)
            wmma::store_matrix_sync(C + (size_t)row0 * d + col0 + j * 8, c_frag[j], d, wmma::mem_row_major);
}

// BMMA v2 shared-memory tiled
#define TILE_K_WORDS 64
#define TILE_K_BITS  (TILE_K_WORDS * 32)
__global__ void bmma_smem_kernel(const unsigned* __restrict__ A,
                                 const unsigned* __restrict__ B,
                                 int* __restrict__ C, int b, int d, int W32)
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
        #pragma unroll
        for (int idx = tid; idx < 64 * TILE_K_WORDS; idx += 256) {
            const int r = idx / TILE_K_WORDS, c = idx % TILE_K_WORDS;
            As[r][c] = __ldg(gA + (size_t)r * W32 + kw + c);
            Bs[r][c] = __ldg(gB + (size_t)r * W32 + kw + c);
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_K_WORDS; k += 4) {
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
    for (int j = 0; j < 8; j++)
        if (active && col0 + j * 8 < d)
            wmma::store_matrix_sync(C + (size_t)row0 * d + col0 + j * 8, c_frag[j], d, wmma::mem_row_major);
}

// bitmap loop fallback (AnyFIM getResFrequencyCudaKernel shape)
__global__ void bitmap_kernel(const unsigned long long* __restrict__ A,
                              const unsigned long long* __restrict__ B,
                              int* __restrict__ C, int b, int d, int W64)
{
    const int i = blockIdx.x;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= d) return;
    const unsigned long long* a = A + (size_t)i * W64;
    const unsigned long long* v = B + (size_t)j * W64;
    unsigned long long s = 0;
    for (int w = 0; w < W64; w++) s += __popcll(__ldg(a + w) & __ldg(v + w));
    C[(size_t)i * d + j] = (int)s;
}

// ------------------------------- dispatcher ---------------------------------
enum KernelChoice { K_BITMAP = 0, K_BMMA_NAIVE = 1, K_BMMA_SMEM = 2 };
static const char* kname(int k) { return k == K_BITMAP ? "bitmap" : (k == K_BMMA_NAIVE ? "BMMA-naive" : "BMMA-smem"); }

static int dispatch(int b, int d, size_t bytesB, bool forceBitmap)
{
    if (forceBitmap) return K_BITMAP;
    if (b < FALLBACK_B) return K_BITMAP;                    // memo risk #1 fallback
    long blocks = (long)((b + 63) / 64) * (d / 64);
    if (blocks < 76) return K_BMMA_NAIVE;                   // underfed SMs
    if (bytesB > (size_t)256 * 1024 * 1024) return K_BMMA_SMEM;
    return K_BMMA_NAIVE;
}

// ------------------------------- data generation -----------------------------
struct Pattern { std::vector<int> items; double freq; };

static unsigned long long xs64(unsigned long long& s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

static void gen_data(std::vector<unsigned>& B, int W32, std::vector<Pattern>& pats)
{
    unsigned long long seed = 0xC0FFEE123456789ULL;
    // patterns: size 2..MAXPAT, freq 0.2%..1.5%
    for (int p = 0; p < N_PAT; p++) {
        Pattern pat;
        int sz = 2 + (int)(xs64(seed) % (MAXPAT - 1));
        pat.freq = (0.2 + 1.3 * (xs64(seed) % 1000) / 1000.0) / 100.0;
        std::set<int> used;
        while ((int)pat.items.size() < sz) {
            int it = (int)(xs64(seed) % D_ITEMS);
            if (used.insert(it).second) pat.items.push_back(it);
        }
        std::sort(pat.items.begin(), pat.items.end());
        pats.push_back(pat);
    }
    // fill bitmaps: iterate transactions, each pattern fires i.i.d. with prob freq
    std::vector<unsigned> thr(N_PAT);
    for (int p = 0; p < N_PAT; p++) thr[p] = (unsigned)(pats[p].freq * 4294967296.0);
    for (size_t t = 0; t < N_TX; t++) {
        unsigned wd = (unsigned)(t >> 5), bit = 1u << (t & 31);
        for (int p = 0; p < N_PAT; p++) {
            if ((unsigned)xs64(seed) < thr[p]) {
                for (int it : pats[p].items) B[(size_t)it * W32 + wd] |= bit;
            }
        }
    }
    // light noise so most items are infrequent
    for (int j = 0; j < D_ITEMS; j++) {
        int n = (int)(N_TX * 0.0002);
        for (int q = 0; q < n; q++) {
            size_t t = xs64(seed) % N_TX;
            B[(size_t)j * W32 + (t >> 5)] |= (1u << (t & 31));
        }
    }
}

// exact host-side support of an itemset (validation only)
static int host_support(const std::vector<unsigned>& B, int W32, const std::vector<int>& items)
{
    int s = 0;
    for (int w = 0; w < W32; w++) {
        unsigned x = ~0u;
        for (int it : items) x &= B[(size_t)it * W32 + w];
        s += __popcnt(x);
    }
    return s;
}

// ------------------------------- mining loop --------------------------------
struct Itemset { int items[MAXK]; int k; int support; };

struct LevelStats { int level, prefixes, found, kernel, batches; double evalMs, maskMs; };

static double now_ms()
{
    static LARGE_INTEGER f, b; static bool init = false;
    if (!init) { QueryPerformanceFrequency(&f); init = true; }
    QueryPerformanceCounter(&b);
    return 1000.0 * b.QuadPart / f.QuadPart;
}

int main(int argc, char** argv)
{
    bool forceBitmap = (argc > 1 && strcmp(argv[1], "--bitmap-only") == 0);
    const int W32 = (int)(N_TX / 32);
    const int W64 = (int)(N_TX / 64);
    const int minsup = (int)(N_TX * MINSUP_PC / 100.0);
    const size_t bytesB = (size_t)D_ITEMS * W32 * 4;

    printf("Batch evaluation prototype: N=%zu tx, d=%d, minsup=%d (%.2f%%), mode=%s\n",
           N_TX, D_ITEMS, minsup, MINSUP_PC, forceBitmap ? "bitmap-only" : "dispatch");
    fflush(stdout);

    // ---- data --------------------------------------------------------------
    double t0 = now_ms();
    std::vector<unsigned> hB((size_t)D_ITEMS * W32, 0u);
    std::vector<Pattern> pats;
    gen_data(hB, W32, pats);
    printf("data gen: %.1f s (%zu planted patterns)\n", (now_ms() - t0) / 1e3, pats.size());
    fflush(stdout);

    unsigned *dA, *dB; int* dC; int* d_items; int* d_cnt;
    CK(cudaMalloc(&dB, bytesB));
    CK(cudaMalloc(&dA, (size_t)BATCH * W32 * 4));
    CK(cudaMalloc(&dC, (size_t)BATCH * D_ITEMS * 4));
    CK(cudaMalloc(&d_items, (size_t)BATCH * MAXK * 4));
    CK(cudaMalloc(&d_cnt, D_ITEMS * 4));
    CK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    // ---- F1 ----------------------------------------------------------------
    item_count_kernel<<<(D_ITEMS + 255) / 256, 256>>>(dB, D_ITEMS, W32, d_cnt);
    std::vector<int> h_cnt(D_ITEMS);
    CK(cudaMemcpy(h_cnt.data(), d_cnt, D_ITEMS * 4, cudaMemcpyDeviceToHost));
    std::vector<Itemset> cur, found;
    for (int j = 0; j < D_ITEMS; j++)
        if (h_cnt[j] >= minsup) { Itemset s; s.k = 1; s.items[0] = j; s.support = h_cnt[j]; cur.push_back(s); }
    for (auto& s : cur) found.push_back(s);
    printf("F1: %zu frequent items\n", cur.size());

    // ---- level-wise loop ----------------------------------------------------
    std::vector<LevelStats> stats;
    bool crossCheckOk = true;
    for (int level = 2; level <= MAXK && !cur.empty(); level++) {
        int total = (int)cur.size();
        double evalMs = 0, maskMs = 0;
        int kernelUsed = -1, nBatches = 0;
        std::vector<Itemset> next;
        bool firstBatch = true;

        for (int bs = 0; bs < total; bs += BATCH) {
            int bcur = std::min(BATCH, total - bs);
            nBatches++;
            // pack item lists (parent prefix = items[0..k-2], extension j > items[k-2])
            std::vector<int> h_items((size_t)bcur * (level - 1));
            for (int i = 0; i < bcur; i++)
                memcpy(&h_items[(size_t)i * (level - 1)], cur[bs + i].items, (level - 1) * sizeof(int));
            float ms;
            CK(cudaMemcpy(d_items, h_items.data(), (size_t)bcur * (level - 1) * 4, cudaMemcpyHostToDevice));
            CK(cudaEventRecord(e0));
            mask_build_kernel<<<bcur, 256>>>(dB, d_items, level - 1, W32, dA);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&ms, e0, e1)); maskMs += ms;

            int kc = dispatch(bcur, D_ITEMS, bytesB, forceBitmap);
            if (kernelUsed < 0) kernelUsed = kc;
            else if (kernelUsed != kc) kernelUsed = 99;   // mixed within level
            CK(cudaEventRecord(e0));
            if (kc == K_BITMAP) {
                dim3 g(bcur, (D_ITEMS + 255) / 256);
                bitmap_kernel<<<g, 256>>>((const unsigned long long*)dA,
                                          (const unsigned long long*)dB, dC, bcur, D_ITEMS, W64);
            } else {
                dim3 g((bcur + 63) / 64, D_ITEMS / 64);
                if (kc == K_BMMA_SMEM) bmma_smem_kernel<<<g, 256>>>(dA, dB, dC, bcur, D_ITEMS, W32);
                else                   bmma_kernel<<<g, 256>>>(dA, dB, dC, bcur, D_ITEMS, W32);
            }
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            CK(cudaGetLastError());
            CK(cudaEventElapsedTime(&ms, e0, e1)); evalMs += ms;

            std::vector<int> hC((size_t)bcur * D_ITEMS);
            CK(cudaMemcpy(hC.data(), dC, (size_t)bcur * D_ITEMS * 4, cudaMemcpyDeviceToHost));

            // per-level cross-check: first batch recomputed by bitmap kernel
            if (firstBatch && kc != K_BITMAP) {
                std::vector<int> hCref((size_t)bcur * D_ITEMS);
                dim3 g(bcur, (D_ITEMS + 255) / 256);
                bitmap_kernel<<<g, 256>>>((const unsigned long long*)dA,
                                          (const unsigned long long*)dB, dC, bcur, D_ITEMS, W64);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(hCref.data(), dC, (size_t)bcur * D_ITEMS * 4, cudaMemcpyDeviceToHost));
                if (hCref != hC) { printf("  [FAIL] level %d cross-check mismatch\n", level); crossCheckOk = false; }
                firstBatch = false;
            }

            // prune + canonical extension
            for (int i = 0; i < bcur; i++) {
                int last = cur[bs + i].items[level - 2];
                for (int j = last + 1; j < D_ITEMS; j++) {
                    int sup = hC[(size_t)i * D_ITEMS + j];
                    if (sup >= minsup) {
                        Itemset s; s.k = level; s.support = sup;
                        memcpy(s.items, cur[bs + i].items, (level - 1) * sizeof(int));
                        s.items[level - 1] = j;
                        next.push_back(s);
                        found.push_back(s);
                    }
                }
            }
        }
        stats.push_back({level, total, (int)next.size(), kernelUsed, nBatches, evalMs, maskMs});
        printf("L%d: prefixes=%5d batches=%d kernel=%-11s mask=%7.1f ms eval=%8.1f ms -> found=%6d\n",
               level, total, nBatches, kernelUsed == 99 ? "mixed" : kname(kernelUsed),
               maskMs, evalMs, (int)next.size());
        fflush(stdout);
        cur.swap(next);
    }

    // ---- validation ----------------------------------------------------------
    // 1) downward closure
    std::set<std::vector<int>> foundSet;
    for (auto& s : found) foundSet.insert(std::vector<int>(s.items, s.items + s.k));
    bool closed = true;
    for (auto& s : found) {
        if (s.k <= 2) continue;
        for (int skip = 0; skip < s.k; skip++) {
            std::vector<int> sub;
            for (int t = 0; t < s.k; t++) if (t != skip) sub.push_back(s.items[t]);
            if (!foundSet.count(sub)) { closed = false; break; }
        }
        if (!closed) break;
    }
    // 2) planted recall: pattern frequent (host-exact) <=> in found set
    int patOK = 0, patMiss = 0, patFalse = 0;
    for (auto& p : pats) {
        int sup = host_support(hB, W32, p.items);
        bool in = foundSet.count(p.items) > 0;
        if (sup >= minsup && in) patOK++;
        else if (sup >= minsup && !in) patMiss++;
        else if (sup < minsup && in) patFalse++;
    }
    double evalTot = 0, maskTot = 0;
    for (auto& s : stats) { evalTot += s.evalMs; maskTot += s.maskMs; }
    printf("\n== summary (%s) ==\n", forceBitmap ? "bitmap-only" : "dispatch");
    printf("total frequent itemsets: %zu\n", found.size());
    printf("eval time: %.1f ms   mask build: %.1f ms\n", evalTot, maskTot);
    printf("cross-check (BMMA vs bitmap): %s\n", crossCheckOk ? "OK" : "FAIL");
    printf("downward closure: %s\n", closed ? "OK" : "FAIL");
    printf("planted patterns: recalled=%d missed=%d false-pos=%d (of %zu)\n",
           patOK, patMiss, patFalse, pats.size());
    return (crossCheckOk && closed && patMiss == 0 && patFalse == 0) ? 0 : 1;
}
