/* Pack each input panel once, compute only upper block pairs, then scatter
 * the requested triangle(s). No cached products or relaxed arithmetic. */
#include "symmetric.h"
#include <arm_neon.h>
#include <arm_sve.h>
#include <stdint.h>
#include <sys/prctl.h>
#include "sve_runtime.h"
#include "sve_interleaved.h"
#include "neon_micro12.h"
#ifndef CAMBLAS_SYMMETRIC_BLOCK
#define CAMBLAS_SYMMETRIC_BLOCK 24
#endif
#ifndef CAMBLAS_SYMMETRIC_K32
#define CAMBLAS_SYMMETRIC_K32 256
#endif
#ifndef CAMBLAS_SYMMETRIC_K64
#define CAMBLAS_SYMMETRIC_K64 128
#endif
#define SB CAMBLAS_SYMMETRIC_BLOCK
#ifndef CAMBLAS_SYMMETRIC_BALANCE
#define CAMBLAS_SYMMETRIC_BALANCE 0
#endif
#ifndef CAMBLAS_SYMMETRIC_VECTOR_PACK
#define CAMBLAS_SYMMETRIC_VECTOR_PACK 0
#endif
#ifndef CAMBLAS_SYMMETRIC_B_GAP_BYTES
#define CAMBLAS_SYMMETRIC_B_GAP_BYTES 0
#endif
#ifndef CAMBLAS_SYMMETRIC_DIAGONAL
#define CAMBLAS_SYMMETRIC_DIAGONAL 0
#endif
#ifndef CAMBLAS_SYMMETRIC_A_PAD
#define CAMBLAS_SYMMETRIC_A_PAD 0
#endif
#ifndef CAMBLAS_SYMMETRIC_MICRO12
#define CAMBLAS_SYMMETRIC_MICRO12 0
#endif
#if CAMBLAS_SYMMETRIC_MICRO12 < 0 || CAMBLAS_SYMMETRIC_MICRO12 > 2
#error "Micro12 mode must be off, both precisions, or FP32 only"
#endif
#if (CAMBLAS_SYMMETRIC_MICRO12 == 1 && CAMBLAS_SYMMETRIC_DIAGONAL) || \
    (CAMBLAS_SYMMETRIC_MICRO12 && CAMBLAS_SYMMETRIC_A_PAD)
#error "Micro12 experiment requires ordinary A and complete padded diagonal tiles"
#endif
#define USE_MICRO12(FP64) (CAMBLAS_SYMMETRIC_MICRO12 && (CAMBLAS_SYMMETRIC_MICRO12 == 1 || !(FP64)))
#define SNR(FP64) (USE_MICRO12(FP64) ? 12 : 8)
#if SB % 8 || (CAMBLAS_SYMMETRIC_MICRO12 && SB % 12)
#error "Symmetric tile must be a multiple of the selected B microgroup"
#endif
#if CAMBLAS_SYMMETRIC_A_PAD && CAMBLAS_MICRO8_ROWS != 3
#error "Padded symmetric A requires the three-vector-row kernel"
#endif
#define AP_ROWS(FP64)                                                                           \
    (CAMBLAS_SYMMETRIC_A_PAD ? ((SB + (FP64 ? 6 : 12) - 1) / (FP64 ? 6 : 12)) * (FP64 ? 8 : 16) \
                             : SB)
#if CAMBLAS_SYMMETRIC_B_GAP_BYTES < 0 || CAMBLAS_SYMMETRIC_B_GAP_BYTES % 64
#error "Symmetric B gap must be a nonnegative cache-line multiple"
#endif
#if SB < 8 || SB > 128 || SB % 8
#error "Symmetric tile must be an eight-column multiple in [8,128]"
#endif
#if CAMBLAS_SYMMETRIC_K32 < 1 || CAMBLAS_SYMMETRIC_K64 < 1
#error "Symmetric reduction blocks must be positive"
#endif
static int symmetric_vl_ready(void)
{
#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS > 0
    return camblas_sve_vl_bits_from_prctl(prctl(PR_SVE_GET_VL)) == 128;
#else
    return svcntb() == 16;
#endif
}
/* Minimax contiguous partition: respect edge-tile work without destroying
 * adjacent A-panel reuse. O(tasks * log(total weight)), no dynamic state. */
static int symmetric_groups(const int *weights, int count, int workers, camblas_task_t *groups)
{
    int low = 0, high = 0;
    for (int i = 0; i < count; i++) {
        if (weights[i] > low)
            low = weights[i];
        high += weights[i];
    }
    while (low < high) {
        int limit = low + (high - low) / 2, used = 1, sum = 0;
        for (int i = 0; i < count; i++) {
            if (sum + weights[i] > limit) {
                ++used;
                sum = 0;
            }
            sum += weights[i];
        }
        if (used > workers)
            low = limit + 1;
        else
            high = limit;
    }
    int start = 0, sum = 0, used = 0;
    for (int i = 0; i < count; i++) {
        if (sum + weights[i] > low) {
            groups[used++] = (camblas_task_t){.i0 = start, .i1 = i};
            start = i;
            sum = 0;
        }
        sum += weights[i];
    }
    groups[used++] = (camblas_task_t){.i0 = start, .i1 = count};
    return used;
}
int camblas_symmetric_bytes(int n, int k, int fp64, size_t *bytes)
{
    if (!bytes || n < 1 || n > 1024 || k < 1 || k > 65536 || (fp64 != 0 && fp64 != 1))
        return -1;
    size_t kc = fp64 ? CAMBLAS_SYMMETRIC_K64 : CAMBLAS_SYMMETRIC_K32;
    size_t nb = ((size_t)n + SB - 1) / SB, kb = ((size_t)k + kc - 1) / kc;
    *bytes = nb * kb * (AP_ROWS(fp64) + SB) * kc * (fp64 ? sizeof(double) : sizeof(float)) +
             CAMBLAS_SYMMETRIC_B_GAP_BYTES;
    return 0;
}
#define DEFINE_DUAL_PACK(S, T, FP64, V, LANES, LOAD, STORE)                                        \
    static void symmetric_dual_pack_##S(char trans, int rows, int ks, int kc, const T *a, int lda, \
                                        T *ap, T *bp)                                              \
    {                                                                                              \
        for (int l = 0; l < ks; l++) {                                                             \
            int i = 0;                                                                             \
            if (CAMBLAS_SYMMETRIC_VECTOR_PACK && trans == 'N') {                                   \
                for (; i + LANES <= rows; i += LANES) {                                            \
                    V x = LOAD(a + (size_t)l * lda + i);                                           \
                    size_t ai = CAMBLAS_SYMMETRIC_A_PAD                                            \
                                    ? (size_t)(i / (3 * LANES)) * (4 * LANES) * kc +               \
                                          (size_t)l * (4 * LANES) + i % (3 * LANES)                \
                                    : (size_t)l * SB + i;                                          \
                    STORE(ap + ai, x);                                                             \
                    STORE(bp + (size_t)(i / SNR(FP64)) * SNR(FP64) * kc + (size_t)l * SNR(FP64) +  \
                              i % SNR(FP64),                                                       \
                          x);                                                                      \
                }                                                                                  \
            }                                                                                      \
            for (; i < SB; i++) {                                                                  \
                T x =                                                                              \
                    i < rows ? a[trans == 'N' ? (size_t)l * lda + i : (size_t)i * lda + l] : (T)0; \
                size_t ai = CAMBLAS_SYMMETRIC_A_PAD                                                \
                                ? (size_t)(i / (3 * LANES)) * (4 * LANES) * kc +                   \
                                      (size_t)l * (4 * LANES) + i % (3 * LANES)                    \
                                : (size_t)l * SB + i;                                              \
                ap[ai] = x;                                                                        \
                bp[(size_t)(i / SNR(FP64)) * SNR(FP64) * kc + (size_t)l * SNR(FP64) +              \
                   i % SNR(FP64)] = x;                                                             \
            }                                                                                      \
        }                                                                                          \
    }
DEFINE_DUAL_PACK(f32, float, 0, float32x4_t, 4, vld1q_f32, vst1q_f32)
DEFINE_DUAL_PACK(f64, double, 1, float64x2_t, 2, vld1q_f64, vst1q_f64)
#undef DEFINE_DUAL_PACK
#define DEFINE_SYMMETRIC(S, T, FP64, KC)                                                           \
    typedef struct {                                                                               \
        char trans;                                                                                \
        int n, k, lda, ldc, nb, kb, upper, full;                                                   \
        T alpha, beta;                                                                             \
        const T *a;                                                                                \
        T *c, *ap, *bp;                                                                            \
        const camblas_task_t *tiles;                                                               \
    } symmetric_##S##_work;                                                                        \
    static void symmetric_##S##_pack(const camblas_task_t *task, void *opaque)                     \
    {                                                                                              \
        symmetric_##S##_work *w = opaque;                                                          \
        for (int slot = task->i0; slot < task->i1; slot++) {                                       \
            int rb = slot / w->kb, pc = slot % w->kb;                                              \
            int row = rb * SB, depth = pc * KC, rows = w->n - row < SB ? w->n - row : SB;          \
            int ks = w->k - depth < KC ? w->k - depth : KC;                                        \
            T *ap = w->ap + (size_t)slot * AP_ROWS(FP64) * KC,                                     \
              *bp = w->bp + (size_t)slot * SB * KC;                                                \
            const T *input = w->a + (w->trans == 'N' ? (size_t)depth * w->lda + row                \
                                                     : (size_t)row * w->lda + depth);              \
            symmetric_dual_pack_##S(w->trans, rows, ks, KC, input, w->lda, ap, bp);                \
        }                                                                                          \
    }                                                                                              \
    static void symmetric_##S##_compute(const camblas_task_t *task, void *opaque)                  \
    {                                                                                              \
        symmetric_##S##_work *w = opaque;                                                          \
        int rb = task->i0, cb = task->j0, row = rb * SB, col = cb * SB;                            \
        int rows = w->n - row < SB ? w->n - row : SB, cols = w->n - col < SB ? w->n - col : SB;    \
        _Alignas(64) T product[SB * SB];                                                           \
        for (int pc = 0; pc < w->kb; pc++) {                                                       \
            int ks = w->k - pc * KC < KC ? w->k - pc * KC : KC;                                    \
            const T *ap = w->ap + ((size_t)rb * w->kb + pc) * AP_ROWS(FP64) * KC;                  \
            const T *bp = w->bp + ((size_t)cb * w->kb + pc) * SB * KC;                             \
            if (USE_MICRO12(FP64)) {                                                               \
                camblas_neon_micro12_##S(rows, cols, ks, ap, SB, bp, KC, product, SB, pc == 0);    \
            } else if (CAMBLAS_SYMMETRIC_DIAGONAL && rb == cb) {                                   \
                for (int j = 0; j < cols; j += 8) {                                                \
                    int height = j + 8 < rows ? j + 8 : rows, width = cols - j < 8 ? cols - j : 8; \
                    camblas_micro8_layout_##S(height, width, ks, (T)1, ap,                         \
                                              CAMBLAS_SYMMETRIC_A_PAD ? KC : SB,                   \
                                              bp + (size_t)j * KC, KC, product + (size_t)j * SB,   \
                                              SB, CAMBLAS_SYMMETRIC_A_PAD ? 2 : 0, pc == 0);       \
                }                                                                                  \
            } else                                                                                 \
                camblas_micro8_layout_##S(rows, cols, ks, (T)1, ap,                                \
                                          CAMBLAS_SYMMETRIC_A_PAD ? KC : SB, bp, KC, product, SB,  \
                                          CAMBLAS_SYMMETRIC_A_PAD ? 2 : 0, pc == 0);               \
        }                                                                                          \
        if (w->upper || w->full)                                                                   \
            for (int j = 0; j < cols; j++) {                                                       \
                int end = rb == cb && j + 1 < rows ? j + 1 : rows;                                 \
                for (int i = 0; i < end; i++) {                                                    \
                    T value = w->alpha * product[(size_t)j * SB + i];                              \
                    size_t at = (size_t)(col + j) * w->ldc + row + i;                              \
                    w->c[at] = w->beta == (T)0 ? value : value + w->beta * w->c[at];               \
                }                                                                                  \
            }                                                                                      \
        if (!w->upper || w->full)                                                                  \
            for (int i = 0; i < rows; i++) {                                                       \
                int start = rb == cb ? i + (w->full != 0) : 0;                                     \
                for (int j = start; j < cols; j++) {                                               \
                    T value = w->alpha * product[(size_t)j * SB + i];                              \
                    size_t at = (size_t)(row + i) * w->ldc + col + j;                              \
                    w->c[at] = w->beta == (T)0 ? value : value + w->beta * w->c[at];               \
                }                                                                                  \
            }                                                                                      \
    }                                                                                              \
    static void symmetric_##S##_group(const camblas_task_t *task, void *opaque)                    \
    {                                                                                              \
        symmetric_##S##_work *w = opaque;                                                          \
        for (int i = task->i0; i < task->i1; i++)                                                  \
            symmetric_##S##_compute(w->tiles + i, opaque);                                         \
    }                                                                                              \
    int camblas_symmetric_##S(const camblas_executor_t *executor, int workers, char trans, int n,  \
                              int k, T alpha, const T *a, int lda, T beta, T *c, int ldc,          \
                              int upper, int full, void *scratch, size_t capacity)                 \
    {                                                                                              \
        size_t needed;                                                                             \
        if (!executor || !executor->run || workers < 1 || workers > 64 || !a || !c || !scratch ||  \
            (trans != 'N' && trans != 'T') || lda < (trans == 'N' ? n : k) || ldc < n ||           \
            camblas_symmetric_bytes(n, k, FP64, &needed) || capacity < needed ||                   \
            !symmetric_vl_ready())                                                                 \
            return -1;                                                                             \
        int nb = (n + SB - 1) / SB, kb = (k + KC - 1) / KC, slots = nb * kb;                       \
        symmetric_##S##_work w = {.trans = trans,                                                  \
                                  .n = n,                                                          \
                                  .k = k,                                                          \
                                  .lda = lda,                                                      \
                                  .ldc = ldc,                                                      \
                                  .nb = nb,                                                        \
                                  .kb = kb,                                                        \
                                  .upper = upper,                                                  \
                                  .full = full,                                                    \
                                  .alpha = alpha,                                                  \
                                  .beta = beta,                                                    \
                                  .a = a,                                                          \
                                  .c = c,                                                          \
                                  .ap = scratch,                                                   \
                                  .bp = (T *)((char *)scratch +                                    \
                                              (size_t)slots * AP_ROWS(FP64) * KC * sizeof(T) +     \
                                              CAMBLAS_SYMMETRIC_B_GAP_BYTES)};                     \
        camblas_task_t packing[256];                                                               \
        int count = 0, chunk = (slots + 255) / 256;                                                \
        for (int start = 0; start < slots; start += chunk)                                         \
            packing[count++] = (camblas_task_t){                                                   \
                .i0 = start, .i1 = start + chunk < slots ? start + chunk : slots};                 \
        if (executor->run(symmetric_##S##_pack, packing, count, &w, executor->user_data))          \
            return -1;                                                                             \
        camblas_task_t tasks[((1024 + SB - 1) / SB) * (((1024 + SB - 1) / SB) + 1) / 2];           \
        count = 0;                                                                                 \
        for (int rb = 0; rb < nb; rb++)                                                            \
            for (int cb = rb; cb < nb; cb++)                                                       \
                tasks[count++] = (camblas_task_t){.i0 = rb, .j0 = cb};                             \
        if (CAMBLAS_SYMMETRIC_BALANCE) {                                                           \
            int weights[sizeof(tasks) / sizeof(tasks[0])];                                         \
            camblas_task_t groups[64];                                                             \
            for (int i = 0; i < count; i++) {                                                      \
                int rows = n - tasks[i].i0 * SB < SB ? n - tasks[i].i0 * SB : SB;                  \
                int cols = n - tasks[i].j0 * SB < SB ? n - tasks[i].j0 * SB : SB;                  \
                weights[i] = rows * cols;                                                          \
                if (CAMBLAS_SYMMETRIC_DIAGONAL && !USE_MICRO12(FP64) &&                            \
                    tasks[i].i0 == tasks[i].j0) {                                                  \
                    weights[i] = 0;                                                                \
                    for (int j = 0; j < cols; j += 8)                                              \
                        weights[i] +=                                                              \
                            (j + 8 < rows ? j + 8 : rows) * (cols - j < 8 ? cols - j : 8);         \
                }                                                                                  \
            }                                                                                      \
            int grouped = symmetric_groups(weights, count, workers, groups);                       \
            w.tiles = tasks;                                                                       \
            return executor->run(symmetric_##S##_group, groups, grouped, &w, executor->user_data); \
        }                                                                                          \
        return executor->run(symmetric_##S##_compute, tasks, count, &w, executor->user_data);      \
    }
DEFINE_SYMMETRIC(f32, float, 0, CAMBLAS_SYMMETRIC_K32)
DEFINE_SYMMETRIC(f64, double, 1, CAMBLAS_SYMMETRIC_K64)
#undef DEFINE_SYMMETRIC
#undef AP_ROWS
#undef SNR
#undef USE_MICRO12
#undef SB
