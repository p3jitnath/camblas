/*
 * CAMBLAS correctness-gated packed blocked executor.
 *
 * This executor packs A and B panels, walks mc/nc/kc blocks, and computes
 * each output element from the packed
 * panels. The SVE library can replace each bounded tile with its hidden SVE
 * hook; ordinary C remains the portable and fail-closed fallback. There is
 * no implicit thread creation in this module; caller-owned executors
 * supply task concurrency, including packing and compute phases.
 *
 * The public GEMM wrapper uses the serial path directly and uses the
 * executor-aware path for caller-owned executors. Depending on the selected
 * policy, workers use private panels or synchronously prepared shared panels.
 */
#include "packed.h"
#include "packing.h"
#include "gemm_bounds.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#ifndef CAMBLAS_SHARED_B_MIN_WIDTH
#define CAMBLAS_SHARED_B_MIN_WIDTH 64
#endif
#ifndef CAMBLAS_SHARED_B_GUARD
#define CAMBLAS_SHARED_B_GUARD 0
#endif
#ifndef CAMBLAS_SHARED_B_DEEP_WIDTH
#define CAMBLAS_SHARED_B_DEEP_WIDTH 0
#endif
#ifndef CAMBLAS_SHARED_B_DEEP_KBLOCKS
#define CAMBLAS_SHARED_B_DEEP_KBLOCKS 0
#endif
#ifndef CAMBLAS_PACKED_FUSED_PREP
#define CAMBLAS_PACKED_FUSED_PREP 0
#endif
#ifndef CAMBLAS_PACKED_PRIVATE_DEEP_A
#define CAMBLAS_PACKED_PRIVATE_DEEP_A 0
#endif
#if CAMBLAS_SHARED_B_DEEP_WIDTH < 0 || CAMBLAS_SHARED_B_DEEP_WIDTH % 8
#error "Deep shared B task width must be zero or a positive multiple of eight"
#endif
#if CAMBLAS_SHARED_B_MIN_WIDTH < 8 || CAMBLAS_SHARED_B_MIN_WIDTH % 8
#error "Shared B task width must be a positive multiple of eight"
#endif
#ifndef CAMBLAS_EXPERIMENTAL_SUM_PACK
#define CAMBLAS_EXPERIMENTAL_SUM_PACK 0
#endif
#ifndef CAMBLAS_PACKED_SHARED_A
#define CAMBLAS_PACKED_SHARED_A 0
#endif
#ifndef CAMBLAS_PACKED_ADAPTIVE_A
#define CAMBLAS_PACKED_ADAPTIVE_A 0
#endif
#ifndef CAMBLAS_PACKED_PRIVATE_K_LIMIT
#define CAMBLAS_PACKED_PRIVATE_K_LIMIT 0
#endif
#ifndef CAMBLAS_PACKED_SUBTILES
#define CAMBLAS_PACKED_SUBTILES 0
#endif
#if CAMBLAS_PACKED_SHARED_A && (CAMBLAS_EXPERIMENTAL_SUM_PACK || CAMBLAS_PACKED_A_MICRO64)
#error "Shared A research path requires ordinary, unsummed A"
#endif
#if CAMBLAS_EXPERIMENTAL_SUM_PACK
#include <arm_neon.h>
#endif
#ifndef CAMBLAS_SUM_PACK_ONE_PASS
#define CAMBLAS_SUM_PACK_ONE_PASS 0
#endif
#if CAMBLAS_EXPERIMENTAL_SUM_PACK && \
    (!CAMBLAS_PACKED_SHARED_B || CAMBLAS_PACKED_SHARED_B_CHUNKS || CAMBLAS_PACKED_A_MICRO64)
#error "Experimental sums require ordinary A and contiguous shared B"
#endif

#if defined(CAMBLAS_PACKED_INIT_C) && CAMBLAS_PACKED_INIT_C
#if !defined(CAMBLAS_PACKED_INTERLEAVED) || !CAMBLAS_PACKED_INTERLEAVED || \
    (defined(CAMBLAS_PACKED_MICRO6) && CAMBLAS_PACKED_MICRO6) ||           \
    (defined(CAMBLAS_PACKED_MICRO6_PAD8) && CAMBLAS_PACKED_MICRO6_PAD8) || \
    (defined(CAMBLAS_PACKED_A_MICRO64) && CAMBLAS_PACKED_A_MICRO64)
#error "Direct C initialization requires ordinary A and B-micro8"
#endif
#endif

static int packed_can_initialize(int nr, int has_product, int beta_zero)
{
#if defined(CAMBLAS_PACKED_INIT_C) && CAMBLAS_PACKED_INIT_C
    return nr == 8 && has_product && beta_zero;
#else
    (void)nr;
    (void)has_product;
    (void)beta_zero;
    return 0;
#endif
}

#if defined(CAMBLAS_PACKED_A_MICRO64) && CAMBLAS_PACKED_A_MICRO64
#if !defined(CAMBLAS_PACKED_INTERLEAVED) || !CAMBLAS_PACKED_INTERLEAVED
#error "A-micro64 requires the B-micro8 packing contract"
#endif
#if (defined(CAMBLAS_PACKED_MICRO6) && CAMBLAS_PACKED_MICRO6) || \
    (defined(CAMBLAS_PACKED_MICRO6_PAD8) && CAMBLAS_PACKED_MICRO6_PAD8)
#error "A-micro64 cannot be combined with B-micro6"
#endif
#endif

#ifndef CAMBLAS_PACKED_PAD_A
#define CAMBLAS_PACKED_PAD_A 0
#endif
#ifndef CAMBLAS_PACKED_PAD_CACHELINE
#define CAMBLAS_PACKED_PAD_CACHELINE 0
#endif
#ifndef CAMBLAS_PACKED_PAD_SET8
#define CAMBLAS_PACKED_PAD_SET8 0
#endif
#ifndef CAMBLAS_PACKED_DEEP_ALLOC_FREE
#define CAMBLAS_PACKED_DEEP_ALLOC_FREE 0
#endif
#ifndef CAMBLAS_PACKED_DEEP_PAD_COPY
#define CAMBLAS_PACKED_DEEP_PAD_COPY 0
#endif
#ifndef CAMBLAS_PACKED_DEEP_AMICRO32
#define CAMBLAS_PACKED_DEEP_AMICRO32 0
#endif
/* Generic builds keep these layouts disabled; Grace selects bounded policies. */
#ifndef CAMBLAS_PACKED_DEEP_AMICRO64
#define CAMBLAS_PACKED_DEEP_AMICRO64 0
#endif
#ifndef CAMBLAS_PACKED_DEEP64_MAX_TASKS
#define CAMBLAS_PACKED_DEEP64_MAX_TASKS 32
#endif
#ifndef CAMBLAS_PACKED_DEEP64_MAX_ASPECT
#define CAMBLAS_PACKED_DEEP64_MAX_ASPECT 4
#endif
#ifndef CAMBLAS_PACKED_SHALLOW_AMICRO32
#define CAMBLAS_PACKED_SHALLOW_AMICRO32 0
#endif
#ifndef CAMBLAS_PACKED_SHALLOW_AMICRO32_WIDE
#define CAMBLAS_PACKED_SHALLOW_AMICRO32_WIDE 0
#endif
#ifndef CAMBLAS_PACKED_SHALLOW_AMICRO32_TRANSPOSE
#define CAMBLAS_PACKED_SHALLOW_AMICRO32_TRANSPOSE 0
#endif
#if CAMBLAS_PACKED_SHALLOW_AMICRO32_TRANSPOSE
#include "amicro_transpose.h"
#endif
#ifndef CAMBLAS_PACKED_A_COPY_DEPTH
#define CAMBLAS_PACKED_A_COPY_DEPTH 16
#endif
#if CAMBLAS_PACKED_A_COPY_DEPTH < 1 || CAMBLAS_PACKED_A_COPY_DEPTH > 64
#error "A packing depth tile must be between one and 64"
#endif
#ifndef CAMBLAS_PACKED_DEEP_VECTOR_B
#define CAMBLAS_PACKED_DEEP_VECTOR_B 0
#endif
#ifndef CAMBLAS_PACKED_DEEP_MIN_TASKS
#define CAMBLAS_PACKED_DEEP_MIN_TASKS 16
#endif
#if CAMBLAS_PACKED_DEEP_VECTOR_B && defined(__aarch64__)
#include <arm_neon.h>
#endif

/* Experimental cache-set dispersion. The alignment remains a multiple of
 * mr; sizing and packing must use the same value. Large caller values retain
 * their original alignment to avoid an overflowing integer multiplication. */
static int a_pack_alignment(int mr, int rows, size_t element_bytes)
{
#if defined(CAMBLAS_PACKED_A_MICRO64) && CAMBLAS_PACKED_A_MICRO64
    /* A-micro groups contain six doubles. Keep preflight capacity rounded
       to both six and MR; other layouts may use this harmless padding. */
    if (element_bytes == 8 && mr > 0 && mr <= INT_MAX / 6)
        return 6 * mr;
#endif
    /* Keep every column cache-line aligned, while dispersing a power-of-two
       stride over more cache sets. This is a Grace-specific experiment.
       Only use it where cache-line padding also preserves the caller's MR.
       Capacity preflight and actual packing call this identical function. */
    if (CAMBLAS_PACKED_PAD_CACHELINE && rows >= 64 && mr > 0 &&
        (element_bytes == 4 || element_bytes == 8)) {
        size_t line_elements = 64 / element_bytes;
        if (line_elements % (size_t)mr == 0) {
            size_t lines = ((size_t)rows + line_elements - 1) / line_elements;
            if ((lines & (lines - 1)) == 0 || (CAMBLAS_PACKED_PAD_SET8 && lines % 8 == 0))
                ++lines;
            size_t leading = lines * line_elements;
            if (leading <= INT_MAX)
                return (int)leading;
        }
    }
    return CAMBLAS_PACKED_PAD_A && mr > 0 && mr <= INT_MAX / 3 ? 3 * mr : mr;
}

static int round_up_size(size_t value, size_t multiple, size_t *out)
{
    size_t remainder;
    size_t add;

    if (!out || multiple == 0)
        return -1;
    remainder = value % multiple;
    if (remainder == 0) {
        *out = value;
        return 0;
    }
    add = multiple - remainder;
    if (value > SIZE_MAX - add)
        return -1;
    *out = value + add;
    return 0;
}

static int checked_product(size_t a, size_t b, size_t *out)
{
    if (!out || (a != 0 && b > SIZE_MAX / a))
        return -1;
    *out = a * b;
    return 0;
}

static int valid_gemm_args(char trans_a, char trans_b, int m, int n, int k, int lda, int ldb,
                           int ldc, const void *A, const void *B, const void *C)
{
    int min_lda, min_ldb;

    if ((trans_a != 'N' && trans_a != 'T') || (trans_b != 'N' && trans_b != 'T') || m < 0 ||
        n < 0 || k < 0 || !A || !B || !C)
        return -1;
    if (m == 0 || n == 0)
        return lda >= 1 && ldb >= 1 && ldc >= 1;

    min_lda = (trans_a == 'N') ? m : k;
    min_ldb = (trans_b == 'N') ? k : n;
    if (min_lda < 1)
        min_lda = 1;
    if (min_ldb < 1)
        min_ldb = 1;
    return lda >= min_lda && ldb >= min_ldb && ldc >= m ? 0 : -1;
}

/* Compute the maximum element count needed for one packed panel. */
static int panel_capacity(int rows, int cols, int row_multiple, int col_multiple, size_t *elements)
{
    size_t padded_rows, padded_cols;

    if (!elements || rows < 1 || cols < 1 || row_multiple < 1 || col_multiple < 1)
        return -1;
    if (round_up_size((size_t)rows, (size_t)row_multiple, &padded_rows) != 0 ||
        round_up_size((size_t)cols, (size_t)col_multiple, &padded_cols) != 0 ||
        checked_product(padded_rows, padded_cols, elements) != 0)
        return -1;
    return 0;
}

/*
 * Compute the exact maximum panel sizes needed by this operation.  This is a
 * size/representability preflight only: the caller-owned executor allocates
 * one private pair of panels after dispatch, inside each task.  Keeping the
 * preflight before executor->run makes an impossible workspace an
 * UNAVAILABLE result with C untouched, while a later task allocation failure
 * remains an EXECUTION_ERROR.
 */
static int b_pack_columns(int nr)
{
#if defined(CAMBLAS_PACKED_MICRO6_PAD8) && CAMBLAS_PACKED_MICRO6_PAD8
    if (nr == 8)
        return 6;
#endif
#if defined(CAMBLAS_PACKED_MICRO6) && CAMBLAS_PACKED_MICRO6
    if (nr == 8)
        return 6;
#endif
    return nr;
}

static int workspace_capacities(int m, int n, int k, int mc, int nc, int kc, int mr, int nr,
                                size_t element_bytes, size_t *a_elements, size_t *b_elements)
{
    int block_m, block_n, block_k;

    if (!a_elements || !b_elements || m < 1 || n < 1 || k < 1 || mc < 1 || nc < 1 || kc < 1 ||
        mr < 1 || nr < 1)
        return -1;
    block_m = m < mc ? m : mc;
    block_n = n < nc ? n : nc;
    block_k = k < kc ? k : kc;
    if (panel_capacity(block_m, block_k, a_pack_alignment(mr, block_m, element_bytes), 1,
                       a_elements) != 0 ||
        panel_capacity(block_k, block_n, 1, b_pack_columns(nr), b_elements) != 0)
        return -1;
#if defined(CAMBLAS_PACKED_MICRO6_PAD8) && CAMBLAS_PACKED_MICRO6_PAD8
    if (nr == 8 && checked_product(*b_elements / 6, 8, b_elements) != 0)
        return -1;
#endif
    return 0;
}

/*
 * Optional execution tracing for the correctness harness.  The normal packed
 * path does not inspect this environment variable per tile: the flag is
 * sampled once per GEMM call, and the counters are only incremented when the
 * caller explicitly requests a trace.  This gives a bounded, actual-execution
 * distinction between a successful SVE packed hook and the ordinary-C
 * fallback without adding a public ABI surface or changing observation
 * records.
 */
typedef struct {
    int enabled;
    int require_sve;
    atomic_ulong sve_tiles;
    atomic_ulong ordinary_tiles;
    atomic_ulong pack_a_panels;
    atomic_ulong pack_b_panels;
    atomic_ulong pack_a_elements;
    atomic_ulong pack_b_elements;
    atomic_ulong compute_tiles;
    atomic_ulong edge_tiles;
    atomic_int required_failure;
} packed_trace_t;

static void packed_trace_init(packed_trace_t *trace)
{
    const char *requested = getenv("CAMBLAS_PACKED_TRACE");
    const char *required = getenv("CAMBLAS_PACKED_REQUIRE_SVE_HOOK");
    int trace_requested = requested && requested[0] == '1' && requested[1] == '\0';
    int require_sve = required && required[0] == '1' && required[1] == '\0';
    if (!trace)
        return;
    trace->enabled = trace_requested || require_sve;
    trace->require_sve = require_sve;
    atomic_init(&trace->sve_tiles, 0);
    atomic_init(&trace->ordinary_tiles, 0);
    atomic_init(&trace->pack_a_panels, 0);
    atomic_init(&trace->pack_b_panels, 0);
    atomic_init(&trace->pack_a_elements, 0);
    atomic_init(&trace->pack_b_elements, 0);
    atomic_init(&trace->compute_tiles, 0);
    atomic_init(&trace->edge_tiles, 0);
    atomic_init(&trace->required_failure, 0);
}

static void packed_trace_note_pack(packed_trace_t *trace, int is_a, size_t elements)
{
    if (!trace || !trace->enabled)
        return;
    if (is_a) {
        (void)atomic_fetch_add_explicit(&trace->pack_a_panels, 1, memory_order_relaxed);
        (void)atomic_fetch_add_explicit(&trace->pack_a_elements, (unsigned long)elements,
                                        memory_order_relaxed);
    } else {
        (void)atomic_fetch_add_explicit(&trace->pack_b_panels, 1, memory_order_relaxed);
        (void)atomic_fetch_add_explicit(&trace->pack_b_elements, (unsigned long)elements,
                                        memory_order_relaxed);
    }
}

static void packed_trace_note_tile(packed_trace_t *trace, int is_edge)
{
    if (!trace || !trace->enabled)
        return;
    (void)atomic_fetch_add_explicit(&trace->compute_tiles, 1, memory_order_relaxed);
    if (is_edge)
        (void)atomic_fetch_add_explicit(&trace->edge_tiles, 1, memory_order_relaxed);
}

static void packed_trace_note(packed_trace_t *trace, int used_sve)
{
    if (!trace || !trace->enabled)
        return;
    if (used_sve)
        (void)atomic_fetch_add_explicit(&trace->sve_tiles, 1, memory_order_relaxed);
    else
        (void)atomic_fetch_add_explicit(&trace->ordinary_tiles, 1, memory_order_relaxed);
    if (!used_sve && trace->require_sve)
        atomic_store_explicit(&trace->required_failure, 1, memory_order_release);
}

static int packed_trace_failed(const packed_trace_t *trace)
{
    return trace && trace->require_sve &&
           atomic_load_explicit(&trace->required_failure, memory_order_acquire) != 0;
}

static void packed_trace_report(const char *precision, const char *executor,
                                const packed_trace_t *trace)
{
    unsigned long sve_tiles, ordinary_tiles;
    unsigned long pack_a_panels, pack_b_panels;
    unsigned long pack_a_elements, pack_b_elements;
    unsigned long compute_tiles, edge_tiles;
    if (!trace || !trace->enabled)
        return;
    sve_tiles = atomic_load_explicit(&trace->sve_tiles, memory_order_relaxed);
    ordinary_tiles = atomic_load_explicit(&trace->ordinary_tiles, memory_order_relaxed);
    pack_a_panels = atomic_load_explicit(&trace->pack_a_panels, memory_order_relaxed);
    pack_b_panels = atomic_load_explicit(&trace->pack_b_panels, memory_order_relaxed);
    pack_a_elements = atomic_load_explicit(&trace->pack_a_elements, memory_order_relaxed);
    pack_b_elements = atomic_load_explicit(&trace->pack_b_elements, memory_order_relaxed);
    compute_tiles = atomic_load_explicit(&trace->compute_tiles, memory_order_relaxed);
    edge_tiles = atomic_load_explicit(&trace->edge_tiles, memory_order_relaxed);
    fprintf(stderr,
            "CAMBLAS_PACKED_TRACE precision=%s executor=%s "
            "pack_a_panels=%lu pack_b_panels=%lu "
            "pack_a_elements=%lu pack_b_elements=%lu "
            "compute_tiles=%lu edge_tiles=%lu "
            "sve_tiles=%lu ordinary_tiles=%lu require_sve=%d "
            "required_failure=%d\n",
            precision, executor, pack_a_panels, pack_b_panels, pack_a_elements, pack_b_elements,
            compute_tiles, edge_tiles, sve_tiles, ordinary_tiles, trace->require_sve,
            packed_trace_failed(trace));
}

static void scale_f32(float *C, int m, int n, int ldc, float beta)
{
    if (beta == 1.0f)
        return;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (beta == 0.0f)
                C[(size_t)i + (size_t)j * (size_t)ldc] = 0.0f;
            else
                C[(size_t)i + (size_t)j * (size_t)ldc] *= beta;
        }
    }
}

static void scale_f64(double *C, int m, int n, int ldc, double beta)
{
    if (beta == 1.0)
        return;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (beta == 0.0)
                C[(size_t)i + (size_t)j * (size_t)ldc] = 0.0;
            else
                C[(size_t)i + (size_t)j * (size_t)ldc] *= beta;
        }
    }
}

static int pack_b_f32(char trans, int rows, int cols, int lda, int row0, int col0, int panel_rows,
                      int panel_cols, int row_multiple, int col_multiple, const float *src,
                      float *dst, size_t dst_elems, camblas_packed_panel_t *shape)
{
#if defined(CAMBLAS_PACKED_MICRO6_PAD8) && CAMBLAS_PACKED_MICRO6_PAD8
    if (row_multiple == 1 && col_multiple == 8)
        return camblas_pack_f32_bpanel_micro6_padded(trans, rows, cols, lda, row0, col0, panel_rows,
                                                     panel_cols, src, dst, dst_elems, shape);
#endif
#if defined(CAMBLAS_PACKED_MICRO6) && CAMBLAS_PACKED_MICRO6
    if (row_multiple == 1 && col_multiple == 8)
        return camblas_pack_f32_bpanel_micro6(trans, rows, cols, lda, row0, col0, panel_rows,
                                              panel_cols, src, dst, dst_elems, shape);
#endif
#if defined(CAMBLAS_PACKED_INTERLEAVED) && CAMBLAS_PACKED_INTERLEAVED
    if (row_multiple == 1 && col_multiple == 8)
        return camblas_pack_f32_bpanel_interleaved(trans, rows, cols, lda, row0, col0, panel_rows,
                                                   panel_cols, src, dst, dst_elems, shape);
#endif
    return camblas_pack_f32_panel(trans, rows, cols, lda, row0, col0, panel_rows, panel_cols,
                                  row_multiple, col_multiple, src, dst, dst_elems, shape);
}

static int compute_f32_tile(const camblas_packed_panel_t *a_shape,
                            const camblas_packed_panel_t *b_shape, int block_m, int block_n,
                            int block_k, int mr, int nr, float alpha, const float *a_panel,
                            const float *b_panel, float *C, int ldc, int ic, int jc,
                            packed_trace_t *trace, int initialize)
{
    /* A packed tile hook and its ordinary-C fallback only accumulate
       alpha*A*B after the caller has applied beta.  No product means no
       panel or C access, including for a direct zero-K caller. */
    if (alpha == 0.0f || block_m == 0 || block_n == 0 || block_k == 0)
        return 0;
    if (a_shape->layout == CAMBLAS_PANEL_A_MICRO) {
        if (a_shape->row_group != 12 || b_shape->layout != CAMBLAS_PANEL_B_MICRO8)
            return CAMBLAS_PACKED_EXECUTION_ERROR;
        int rc = CAMBLAS_PACKED_UNAVAILABLE;
        if (camblas_sgemm_sve_amicro12_tile)
            rc = camblas_sgemm_sve_amicro12_tile(
                block_m, block_n, block_k, alpha, a_panel, a_shape->ld, b_panel, b_shape->ld,
                C + (size_t)ic + (size_t)jc * ldc, ldc, initialize);
        if (rc != 0 && rc != CAMBLAS_PACKED_UNAVAILABLE)
            return CAMBLAS_PACKED_EXECUTION_ERROR;
        if (rc == CAMBLAS_PACKED_UNAVAILABLE) {
            for (size_t j = 0; j < (size_t)block_n; j++)
                for (size_t i = 0; i < (size_t)block_m; i++) {
                    float acc = 0;
                    for (size_t q = 0; q < (size_t)block_k; q++)
                        acc += a_panel[(i / 12 * 12) * a_shape->ld + q * 12 + i % 12] *
                               b_panel[(j / 8 * 8) * b_shape->ld + q * 8 + j % 8];
                    float *out = C + (size_t)ic + i + ((size_t)jc + j) * ldc;
                    if (initialize)
                        *out = alpha * acc;
                    else
                        *out += alpha * acc;
                }
        }
        if (trace && trace->enabled) {
            for (size_t j = 0; j < (size_t)block_n; j += (size_t)nr)
                for (size_t i = 0; i < (size_t)block_m; i += (size_t)mr) {
                    packed_trace_note_tile(trace, i + (size_t)mr > (size_t)block_m ||
                                                      j + (size_t)nr > (size_t)block_n);
                    packed_trace_note(trace, rc == 0);
                }
        }
        return 0;
    }
    if (initialize && (a_shape->layout != CAMBLAS_PANEL_COLUMN_MAJOR ||
                       b_shape->layout != CAMBLAS_PANEL_B_MICRO8))
        return CAMBLAS_PACKED_EXECUTION_ERROR;
    if (b_shape->layout == CAMBLAS_PANEL_B_MICRO8 || b_shape->layout == CAMBLAS_PANEL_B_MICRO6 ||
        b_shape->layout == CAMBLAS_PANEL_B_MICRO6_PAD8) {
        int padded6 = b_shape->layout == CAMBLAS_PANEL_B_MICRO6_PAD8;
        size_t width = b_shape->layout == CAMBLAS_PANEL_B_MICRO8 ? 8 : 6;
        size_t stride = padded6 ? 8 : width;
        int (*hook)(int, int, int, float, const float *, int, const float *, int, float *, int) =
            initialize ? camblas_sgemm_sve_interleaved_init_tile
            : padded6
                ? camblas_sgemm_sve_micro6_padded_tile
                : (width == 6 ? camblas_sgemm_sve_micro6_tile : camblas_sgemm_sve_interleaved_tile);
        int rc = CAMBLAS_PACKED_UNAVAILABLE;
        if (hook)
            rc = hook(block_m, block_n, block_k, alpha, a_panel, a_shape->ld, b_panel, b_shape->ld,
                      C + (size_t)ic + (size_t)jc * ldc, ldc);
        if (rc != 0 && rc != CAMBLAS_PACKED_UNAVAILABLE)
            return CAMBLAS_PACKED_EXECUTION_ERROR;
        if (rc == CAMBLAS_PACKED_UNAVAILABLE) {
            for (size_t j = 0; j < (size_t)block_n; j++)
                for (size_t i = 0; i < (size_t)block_m; i++) {
                    float acc = 0;
                    for (size_t l = 0; l < (size_t)block_k; l++)
                        acc += a_panel[i + l * a_shape->ld] *
                               b_panel[(j / width * stride) * b_shape->ld + l * stride + j % width];
                    if (initialize)
                        C[(size_t)ic + i + ((size_t)jc + j) * ldc] = alpha * acc;
                    else
                        C[(size_t)ic + i + ((size_t)jc + j) * ldc] += alpha * acc;
                }
        }
        if (!trace || !trace->enabled)
            return 0;
        for (size_t j = 0; j < (size_t)block_n; j += (size_t)nr)
            for (size_t i = 0; i < (size_t)block_m; i += (size_t)mr) {
                packed_trace_note_tile(trace, i + (size_t)mr > (size_t)block_m ||
                                                  j + (size_t)nr > (size_t)block_n);
                packed_trace_note(trace, rc == 0);
            }
        return 0;
    }
    if (b_shape->layout != CAMBLAS_PANEL_COLUMN_MAJOR)
        return CAMBLAS_PACKED_EXECUTION_ERROR;
    /* The packed SVE body already handles an arbitrary macro-panel shape.
       Try that shape once before entering the legacy microtile loop so the
       caller amortizes the hidden hook boundary over the complete mc x nc
       block.  UNAVAILABLE alone preserves the per-microtile fallback; any
       other status remains a fail-closed execution error. */
    {
        int macro_hook_rc = CAMBLAS_PACKED_UNAVAILABLE;
        int macro_shape_ok =
            (size_t)block_m >= 2U * (size_t)mr && (size_t)block_n >= 8U * (size_t)nr;
        if (macro_shape_ok && camblas_sgemm_sve_packed_tile)
            macro_hook_rc = camblas_sgemm_sve_packed_tile(
                block_m, block_n, block_k, alpha, a_panel, a_shape->ld, b_panel, b_shape->ld,
                C + (size_t)ic + (size_t)jc * (size_t)ldc, ldc);
        if (macro_hook_rc == 0) {
            if (!trace || !trace->enabled)
                return 0;
            for (size_t j0 = 0; j0 < (size_t)block_n; j0 += (size_t)nr) {
                size_t j_end = j0 + (size_t)nr;
                if (j_end > (size_t)block_n)
                    j_end = (size_t)block_n;
                for (size_t i0 = 0; i0 < (size_t)block_m; i0 += (size_t)mr) {
                    size_t i_end = i0 + (size_t)mr;
                    if (i_end > (size_t)block_m)
                        i_end = (size_t)block_m;
                    packed_trace_note_tile(trace,
                                           (i_end - i0) < (size_t)mr || (j_end - j0) < (size_t)nr);
                    packed_trace_note(trace, 1);
                }
            }
            return 0;
        }
        if (macro_hook_rc != CAMBLAS_PACKED_UNAVAILABLE)
            return CAMBLAS_PACKED_EXECUTION_ERROR;
    }
    for (size_t j0 = 0; j0 < (size_t)block_n; j0 += (size_t)nr) {
        size_t j_end = j0 + (size_t)nr;
        if (j_end > (size_t)block_n)
            j_end = (size_t)block_n;
        for (size_t i0 = 0; i0 < (size_t)block_m; i0 += (size_t)mr) {
            size_t i_end = i0 + (size_t)mr;
            if (i_end > (size_t)block_m)
                i_end = (size_t)block_m;
            packed_trace_note_tile(trace, (i_end - i0) < (size_t)mr || (j_end - j0) < (size_t)nr);
            int hook_rc = CAMBLAS_PACKED_UNAVAILABLE;
            if (camblas_sgemm_sve_packed_tile)
                hook_rc = camblas_sgemm_sve_packed_tile(
                    (int)(i_end - i0), (int)(j_end - j0), block_k, alpha, a_panel + i0, a_shape->ld,
                    b_panel + j0 * (size_t)b_shape->ld, b_shape->ld,
                    C + (size_t)ic + i0 + ((size_t)jc + j0) * (size_t)ldc, ldc);
            if (hook_rc == 0) {
                packed_trace_note(trace, 1);
                continue;
            }
            /* Only UNAVAILABLE promises that the hook left this tile
               untouched.  Any other status is an execution error: do not
               run ordinary C over a possibly partial hook result. */
            if (hook_rc != CAMBLAS_PACKED_UNAVAILABLE)
                return CAMBLAS_PACKED_EXECUTION_ERROR;
            packed_trace_note(trace, 0);
            for (size_t j = j0; j < j_end; j++) {
                for (size_t i = i0; i < i_end; i++) {
                    float acc = 0.0f;
                    for (int l = 0; l < block_k; l++) {
                        acc += a_panel[i + (size_t)l * a_shape->ld] *
                               b_panel[(size_t)l + j * b_shape->ld];
                    }
                    C[(size_t)ic + i + ((size_t)jc + j) * (size_t)ldc] += alpha * acc;
                }
            }
        }
    }
    return 0;
}

static int pack_b_f64(char trans, int rows, int cols, int lda, int row0, int col0, int panel_rows,
                      int panel_cols, int row_multiple, int col_multiple, const double *src,
                      double *dst, size_t dst_elems, camblas_packed_panel_t *shape)
{
#if defined(CAMBLAS_PACKED_MICRO6_PAD8) && CAMBLAS_PACKED_MICRO6_PAD8
    if (row_multiple == 1 && col_multiple == 8)
        return camblas_pack_f64_bpanel_micro6_padded(trans, rows, cols, lda, row0, col0, panel_rows,
                                                     panel_cols, src, dst, dst_elems, shape);
#endif
#if defined(CAMBLAS_PACKED_MICRO6) && CAMBLAS_PACKED_MICRO6
    if (row_multiple == 1 && col_multiple == 8)
        return camblas_pack_f64_bpanel_micro6(trans, rows, cols, lda, row0, col0, panel_rows,
                                              panel_cols, src, dst, dst_elems, shape);
#endif
#if defined(CAMBLAS_PACKED_INTERLEAVED) && CAMBLAS_PACKED_INTERLEAVED
    if (row_multiple == 1 && col_multiple == 8)
        return camblas_pack_f64_bpanel_interleaved(trans, rows, cols, lda, row0, col0, panel_rows,
                                                   panel_cols, src, dst, dst_elems, shape);
#endif
    return camblas_pack_f64_panel(trans, rows, cols, lda, row0, col0, panel_rows, panel_cols,
                                  row_multiple, col_multiple, src, dst, dst_elems, shape);
}

static int pack_a_f64(int nr, char trans, int rows, int cols, int lda, int row0, int col0,
                      int panel_rows, int panel_cols, int row_multiple, int col_multiple,
                      const double *src, double *dst, size_t dst_elems,
                      camblas_packed_panel_t *shape)
{
#if defined(CAMBLAS_PACKED_A_MICRO64) && CAMBLAS_PACKED_A_MICRO64
    if (nr == 8 && row_multiple % 6 == 0 && col_multiple == 1)
        return camblas_pack_f64_apanel_micro(trans, rows, cols, lda, row0, col0, panel_rows,
                                             panel_cols, 6, src, dst, dst_elems, shape);
#else
    (void)nr;
#endif
    return camblas_pack_f64_panel(trans, rows, cols, lda, row0, col0, panel_rows, panel_cols,
                                  row_multiple, col_multiple, src, dst, dst_elems, shape);
}

static int compute_f64_tile(const camblas_packed_panel_t *a_shape,
                            const camblas_packed_panel_t *b_shape, int block_m, int block_n,
                            int block_k, int mr, int nr, double alpha, const double *a_panel,
                            const double *b_panel, double *C, int ldc, int ic, int jc,
                            packed_trace_t *trace, int initialize)
{
    /* Keep the caller-owned fallback aligned with the hidden hook: an empty
       product is a complete no-op and must not read panels or touch C. */
    if (alpha == 0.0 || block_m == 0 || block_n == 0 || block_k == 0)
        return 0;
    if (initialize && ((a_shape->layout != CAMBLAS_PANEL_COLUMN_MAJOR &&
                        a_shape->layout != CAMBLAS_PANEL_A_MICRO) ||
                       b_shape->layout != CAMBLAS_PANEL_B_MICRO8))
        return CAMBLAS_PACKED_EXECUTION_ERROR;
    if (a_shape->layout == CAMBLAS_PANEL_A_MICRO) {
        if (b_shape->layout != CAMBLAS_PANEL_B_MICRO8 || a_shape->row_group != 6)
            return CAMBLAS_PACKED_EXECUTION_ERROR;
        int rc = CAMBLAS_PACKED_UNAVAILABLE;
        if (camblas_dgemm_sve_amicro6_tile)
            rc = camblas_dgemm_sve_amicro6_tile(block_m, block_n, block_k, alpha, a_panel,
                                                a_shape->ld, b_panel, b_shape->ld,
                                                C + (size_t)ic + (size_t)jc * ldc, ldc, initialize);
        else if (!initialize && camblas_dgemm_sve_amicro_tile)
            rc = camblas_dgemm_sve_amicro_tile(block_m, block_n, block_k, alpha, a_panel,
                                               a_shape->ld, b_panel, b_shape->ld,
                                               C + (size_t)ic + (size_t)jc * ldc, ldc);
        if (rc != 0 && rc != CAMBLAS_PACKED_UNAVAILABLE)
            return CAMBLAS_PACKED_EXECUTION_ERROR;
        if (rc == CAMBLAS_PACKED_UNAVAILABLE) {
            for (size_t j = 0; j < (size_t)block_n; j++)
                for (size_t i = 0; i < (size_t)block_m; i++) {
                    double acc = 0;
                    for (size_t l = 0; l < (size_t)block_k; l++)
                        acc += a_panel[(i / 6 * 6) * a_shape->ld + l * 6 + i % 6] *
                               b_panel[(j / 8 * 8) * b_shape->ld + l * 8 + j % 8];
                    double *out = C + (size_t)ic + i + ((size_t)jc + j) * ldc;
                    if (initialize)
                        *out = alpha * acc;
                    else
                        *out += alpha * acc;
                }
        }
        if (!trace || !trace->enabled)
            return 0;
        for (size_t j = 0; j < (size_t)block_n; j += (size_t)nr)
            for (size_t i = 0; i < (size_t)block_m; i += (size_t)mr) {
                packed_trace_note_tile(trace, i + (size_t)mr > (size_t)block_m ||
                                                  j + (size_t)nr > (size_t)block_n);
                packed_trace_note(trace, rc == 0);
            }
        return 0;
    }
    if (a_shape->layout != CAMBLAS_PANEL_COLUMN_MAJOR)
        return CAMBLAS_PACKED_EXECUTION_ERROR;
    if (b_shape->layout == CAMBLAS_PANEL_B_MICRO8 || b_shape->layout == CAMBLAS_PANEL_B_MICRO6 ||
        b_shape->layout == CAMBLAS_PANEL_B_MICRO6_PAD8) {
        int padded6 = b_shape->layout == CAMBLAS_PANEL_B_MICRO6_PAD8;
        size_t width = b_shape->layout == CAMBLAS_PANEL_B_MICRO8 ? 8 : 6;
        size_t stride = padded6 ? 8 : width;
        int (*hook)(int, int, int, double, const double *, int, const double *, int, double *,
                    int) = initialize ? camblas_dgemm_sve_interleaved_init_tile
                           : padded6  ? camblas_dgemm_sve_micro6_padded_tile
                                      : (width == 6 ? camblas_dgemm_sve_micro6_tile
                                                    : camblas_dgemm_sve_interleaved_tile);
        int rc = CAMBLAS_PACKED_UNAVAILABLE;
        if (hook)
            rc = hook(block_m, block_n, block_k, alpha, a_panel, a_shape->ld, b_panel, b_shape->ld,
                      C + (size_t)ic + (size_t)jc * ldc, ldc);
        if (rc != 0 && rc != CAMBLAS_PACKED_UNAVAILABLE)
            return CAMBLAS_PACKED_EXECUTION_ERROR;
        if (rc == CAMBLAS_PACKED_UNAVAILABLE) {
            for (size_t j = 0; j < (size_t)block_n; j++)
                for (size_t i = 0; i < (size_t)block_m; i++) {
                    double acc = 0;
                    for (size_t l = 0; l < (size_t)block_k; l++)
                        acc += a_panel[i + l * a_shape->ld] *
                               b_panel[(j / width * stride) * b_shape->ld + l * stride + j % width];
                    if (initialize)
                        C[(size_t)ic + i + ((size_t)jc + j) * ldc] = alpha * acc;
                    else
                        C[(size_t)ic + i + ((size_t)jc + j) * ldc] += alpha * acc;
                }
        }
        if (!trace || !trace->enabled)
            return 0;
        for (size_t j = 0; j < (size_t)block_n; j += (size_t)nr)
            for (size_t i = 0; i < (size_t)block_m; i += (size_t)mr) {
                packed_trace_note_tile(trace, i + (size_t)mr > (size_t)block_m ||
                                                  j + (size_t)nr > (size_t)block_n);
                packed_trace_note(trace, rc == 0);
            }
        return 0;
    }
    if (b_shape->layout != CAMBLAS_PANEL_COLUMN_MAJOR)
        return CAMBLAS_PACKED_EXECUTION_ERROR;
    /* Amortize the hidden packed-hook boundary over one complete mc x nc
       panel.  Preserve the existing per-microtile ordinary-C fallback when
       the macro-shaped hook is unavailable, and fail closed on other status. */
    {
        int macro_hook_rc = CAMBLAS_PACKED_UNAVAILABLE;
        int macro_shape_ok =
            (size_t)block_m >= 2U * (size_t)mr && (size_t)block_n >= 8U * (size_t)nr;
        if (macro_shape_ok && camblas_dgemm_sve_packed_tile)
            macro_hook_rc = camblas_dgemm_sve_packed_tile(
                block_m, block_n, block_k, alpha, a_panel, a_shape->ld, b_panel, b_shape->ld,
                C + (size_t)ic + (size_t)jc * (size_t)ldc, ldc);
        if (macro_hook_rc == 0) {
            if (!trace || !trace->enabled)
                return 0;
            for (size_t j0 = 0; j0 < (size_t)block_n; j0 += (size_t)nr) {
                size_t j_end = j0 + (size_t)nr;
                if (j_end > (size_t)block_n)
                    j_end = (size_t)block_n;
                for (size_t i0 = 0; i0 < (size_t)block_m; i0 += (size_t)mr) {
                    size_t i_end = i0 + (size_t)mr;
                    if (i_end > (size_t)block_m)
                        i_end = (size_t)block_m;
                    packed_trace_note_tile(trace,
                                           (i_end - i0) < (size_t)mr || (j_end - j0) < (size_t)nr);
                    packed_trace_note(trace, 1);
                }
            }
            return 0;
        }
        if (macro_hook_rc != CAMBLAS_PACKED_UNAVAILABLE)
            return CAMBLAS_PACKED_EXECUTION_ERROR;
    }
    for (size_t j0 = 0; j0 < (size_t)block_n; j0 += (size_t)nr) {
        size_t j_end = j0 + (size_t)nr;
        if (j_end > (size_t)block_n)
            j_end = (size_t)block_n;
        for (size_t i0 = 0; i0 < (size_t)block_m; i0 += (size_t)mr) {
            size_t i_end = i0 + (size_t)mr;
            if (i_end > (size_t)block_m)
                i_end = (size_t)block_m;
            packed_trace_note_tile(trace, (i_end - i0) < (size_t)mr || (j_end - j0) < (size_t)nr);
            int hook_rc = CAMBLAS_PACKED_UNAVAILABLE;
            if (camblas_dgemm_sve_packed_tile)
                hook_rc = camblas_dgemm_sve_packed_tile(
                    (int)(i_end - i0), (int)(j_end - j0), block_k, alpha, a_panel + i0, a_shape->ld,
                    b_panel + j0 * (size_t)b_shape->ld, b_shape->ld,
                    C + (size_t)ic + i0 + ((size_t)jc + j0) * (size_t)ldc, ldc);
            if (hook_rc == 0) {
                packed_trace_note(trace, 1);
                continue;
            }
            /* A non-UNAVAILABLE status means the hook may have started
               writing C.  Propagate it instead of applying the ordinary-C
               tile a second time. */
            if (hook_rc != CAMBLAS_PACKED_UNAVAILABLE)
                return CAMBLAS_PACKED_EXECUTION_ERROR;
            packed_trace_note(trace, 0);
            for (size_t j = j0; j < j_end; j++) {
                for (size_t i = i0; i < i_end; i++) {
                    double acc = 0.0;
                    for (int l = 0; l < block_k; l++) {
                        acc += a_panel[i + (size_t)l * a_shape->ld] *
                               b_panel[(size_t)l + j * b_shape->ld];
                    }
                    C[(size_t)ic + i + ((size_t)jc + j) * (size_t)ldc] += alpha * acc;
                }
            }
        }
    }
    return 0;
}

/* Keep the task array bounded even when a caller supplies a very large shape. */
#define CAMBLAS_PACKED_MAX_TASKS 256

static int ceil_div_positive(int value, int divisor)
{
    return value / divisor + (value % divisor != 0);
}

static void release_shared_b(void *base, int k, int kc)
{
#if defined(CAMBLAS_PACKED_SHARED_B_CHUNKS) && CAMBLAS_PACKED_SHARED_B_CHUNKS
    if (base) {
        void **panels = base;
        int count = ceil_div_positive(k, kc);
        for (int i = 0; i < count; ++i)
            free(panels[i]);
    }
#else
    (void)k;
    (void)kc;
#endif
    free(base);
}

static void *shared_b_at(void *base, size_t npad, int pc, int kc, size_t size)
{
#if defined(CAMBLAS_PACKED_SHARED_B_CHUNKS) && CAMBLAS_PACKED_SHARED_B_CHUNKS
    (void)npad;
    (void)size;
    return ((void **)base)[pc / kc];
#else
    (void)kc;
    return (char *)base + (size_t)pc * npad * size;
#endif
}

#if defined(CAMBLAS_PACKED_SHARED_B) && CAMBLAS_PACKED_SHARED_B
static void *allocate_shared_b(size_t npad, int k, int kc, size_t size)
{
    size_t elements, bytes;
    if (checked_product(npad, (size_t)k, &elements) || checked_product(elements, size, &bytes) ||
        bytes > PTRDIFF_MAX)
        return NULL;
#if defined(CAMBLAS_PACKED_SHARED_B_CHUNKS) && CAMBLAS_PACKED_SHARED_B_CHUNKS
    int count = ceil_div_positive(k, kc);
    size_t pointer_bytes;
    if (checked_product((size_t)count, sizeof(void *), &pointer_bytes) ||
        pointer_bytes > PTRDIFF_MAX)
        return NULL;
    void **panels = calloc((size_t)count, sizeof(void *));
    if (!panels)
        return NULL;
    for (int i = 0, pc = 0; i < count; ++i) {
        int bk = k - pc < kc ? k - pc : kc;
        panels[i] = malloc(npad * (size_t)bk * size);
        if (!panels[i]) {
            release_shared_b(panels, k, kc);
            return NULL;
        }
        pc += bk;
    }
    return panels;
#else
    (void)kc;
    return malloc(bytes);
#endif
}
#endif

/*
 * Build a bounded 2-D grid of disjoint output rectangles. Each rectangle is
 * an integral group of the planned mc x nc macro tiles. At most sixteen
 * groups are emitted in each dimension, so the fixed task array is enough
 * for every non-negative int shape without an integer-overflowing m+tile-1
 * expression.
 */
static int split_packed_tasks(int m, int n, int mc, int nc, camblas_task_t *tasks, int max_tasks)
{
    int row_blocks, col_blocks, rows_per_task, cols_per_task;
    int n_tasks = 0;

    if (!tasks || max_tasks < 1 || m < 1 || n < 1 || mc < 1 || nc < 1)
        return -1;
    row_blocks = ceil_div_positive(m, mc);
    col_blocks = ceil_div_positive(n, nc);
    rows_per_task = ceil_div_positive(row_blocks, 16);
    cols_per_task = ceil_div_positive(col_blocks, 16);

    for (size_t rb = 0; rb < (size_t)row_blocks; rb += (size_t)rows_per_task) {
        size_t rb_end = rb + (size_t)rows_per_task;
        if (rb_end > (size_t)row_blocks)
            rb_end = (size_t)row_blocks;
        size_t i0 = rb * (size_t)mc;
        size_t i1 = rb_end * (size_t)mc;
        if (i1 > (size_t)m)
            i1 = (size_t)m;

        for (size_t cb = 0; cb < (size_t)col_blocks; cb += (size_t)cols_per_task) {
            size_t cb_end = cb + (size_t)cols_per_task;
            if (cb_end > (size_t)col_blocks)
                cb_end = (size_t)col_blocks;
            size_t j0 = cb * (size_t)nc;
            size_t j1 = cb_end * (size_t)nc;
            if (j1 > (size_t)n)
                j1 = (size_t)n;
            if (n_tasks >= max_tasks || i0 >= i1 || j0 >= j1)
                return -1;
            tasks[n_tasks++] = (camblas_task_t){
                .i0 = (int)i0,
                .i1 = (int)i1,
                .j0 = (int)j0,
                .j1 = (int)j1,
            };
        }
    }
    return n_tasks;
}

typedef struct {
    char trans_a, trans_b;
    int m, n, k, lda, ldb, ldc;
    int mc, nc, kc, mr, nr, n_tasks;
    float alpha, beta;
    const float *A, *B;
#if CAMBLAS_EXPERIMENTAL_SUM_PACK
    const float *A2, *B2;
    int lda2, ldb2, sign_a, sign_b;
#endif
    float *C;
    size_t a_elements, b_elements, a_bytes, b_bytes;
#if CAMBLAS_PACKED_SHARED_A
    float *shared_a;
    camblas_packed_panel_t *shared_a_shapes;
#if CAMBLAS_PACKED_FUSED_PREP
    camblas_task_t *prepared_a;
    int prepared_a_count;
#endif
    int shared_a_kblocks;
    int shared_a_owned;
#endif
    void *shared_b;
#if CAMBLAS_PACKED_FUSED_PREP
    camblas_task_t *prepared_b;
    int prepared_b_count;
#endif
    size_t shared_npad;
    void *borrowed_b;
    size_t borrowed_bytes;
    atomic_int failed;
    packed_trace_t trace;
} sgemm_packed_work_t;

typedef struct {
    char trans_a, trans_b;
    int m, n, k, lda, ldb, ldc;
    int mc, nc, kc, mr, nr, n_tasks;
    double alpha, beta;
    const double *A, *B;
#if CAMBLAS_EXPERIMENTAL_SUM_PACK
    const double *A2, *B2;
    int lda2, ldb2, sign_a, sign_b;
#endif
    double *C;
    size_t a_elements, b_elements, a_bytes, b_bytes;
#if CAMBLAS_PACKED_SHARED_A
    double *shared_a;
    camblas_packed_panel_t *shared_a_shapes;
    int shared_a_kblocks;
    int shared_a_owned;
#endif
    void *shared_b;
    size_t shared_npad;
    void *borrowed_b;
    size_t borrowed_bytes;
    atomic_int failed;
    packed_trace_t trace;
} dgemm_packed_work_t;

#if CAMBLAS_PACKED_SHARED_A
#if CAMBLAS_PACKED_FUSED_PREP
static int sgemm_defer_shared_a(sgemm_packed_work_t *w, const camblas_task_t *tasks, int count)
{
    if (!w->prepared_a)
        return 0;
    for (int i = 0; i < count; i++)
        w->prepared_a[i] = tasks[i];
    w->prepared_a_count = count;
    return 1;
}
#else
#define sgemm_defer_shared_a(w, tasks, count) 0
#endif
/* No new state or branch in the unmodified FP64 packing path. */
#define dgemm_defer_shared_a(w, tasks, count) 0
#define sgemm_private_deep_a(w)                                                           \
    (CAMBLAS_PACKED_PRIVATE_DEEP_A && w->trans_a == 'N' && w->m > w->n && w->m <= 8192 && \
     w->n <= 1024 && w->k > 1024 && w->k <= 4096 && w->n_tasks >= 16 &&                   \
     ceil_div_positive(w->n, w->nc) <= (CAMBLAS_PACKED_PRIVATE_DEEP_A == 1 ? 1 : 4))
#define dgemm_private_deep_a(w) 0
static camblas_packed_panel_t *sgemm_shared_a_descriptors(sgemm_packed_work_t *w, size_t storage,
                                                          size_t bytes)
{
    if (CAMBLAS_PACKED_DEEP_ALLOC_FREE && w->m > w->n && w->m <= 8192 && w->n <= 2048 &&
        w->k > 1024 && w->k <= 4096 && w->n_tasks >= 16 && w->borrowed_b && w->shared_a &&
        !w->shared_a_owned) {
        size_t offset = (size_t)((unsigned char *)w->shared_a - (unsigned char *)w->borrowed_b);
        size_t alignment = _Alignof(camblas_packed_panel_t);
        if (offset <= w->borrowed_bytes && storage <= w->borrowed_bytes - offset) {
            offset += storage;
            size_t pad = (alignment - offset % alignment) % alignment;
            if (pad <= w->borrowed_bytes - offset) {
                offset += pad;
                if (bytes <= w->borrowed_bytes - offset)
                    return (camblas_packed_panel_t *)((unsigned char *)w->borrowed_b + offset);
            }
        }
    }
    return malloc(bytes);
}
static void sgemm_release_a_descriptors(sgemm_packed_work_t *w)
{
    uintptr_t shape = (uintptr_t)w->shared_a_shapes, base = (uintptr_t)w->borrowed_b;
    if (!CAMBLAS_PACKED_DEEP_ALLOC_FREE || !base || shape < base ||
        shape - base >= w->borrowed_bytes)
        free(w->shared_a_shapes);
}
#define dgemm_shared_a_descriptors(w, storage, bytes) malloc(bytes)
static int sgemm_copy_shared_a(sgemm_packed_work_t *w, int ic, int pc, int bm, int bk, float *dst,
                               camblas_packed_panel_t *shape)
{
    int alignment = a_pack_alignment(w->mr, bm, sizeof(float));
    int shallow = CAMBLAS_PACKED_SHALLOW_AMICRO32 &&
                  (w->m >= w->n || (CAMBLAS_PACKED_SHALLOW_AMICRO32_WIDE && w->n_tasks <= 32)) &&
                  w->m >= 128 && w->m <= 2048 && w->n >= 128 && w->n <= 2048 && w->k >= 128 &&
                  w->k <= 1024 && w->n_tasks >= 16 && w->n_tasks <= 64;
    int deep = CAMBLAS_PACKED_DEEP_AMICRO32 && w->m > w->n && w->m <= 8192 && w->n <= 2048 &&
               w->k > 1024 && w->k <= 4096 && w->n_tasks >= CAMBLAS_PACKED_DEEP_MIN_TASKS;
    if ((w->trans_a == 'N' ||
         (shallow && CAMBLAS_PACKED_SHALLOW_AMICRO32_TRANSPOSE && w->n_tasks <= 32)) &&
        w->nr == 8 && (deep || shallow)) {
        camblas_packed_panel_t result;
        if (camblas_packed_panel_shape(w->m, w->k, ic, pc, bm, bk, 12, 1, &result))
            return -1;
        /* Some irregular row blocks need more micro-panel padding than the
         * established capacity. Preserve the ordinary panel in that case. */
        if (result.elements <= w->a_elements) {
            result.layout = CAMBLAS_PANEL_A_MICRO;
            result.row_group = 12;
            result.ld = (size_t)bk;
#if CAMBLAS_PACKED_SHALLOW_AMICRO32_TRANSPOSE
            if (w->trans_a != 'N') {
                camblas_pack_transposed_amicro12(w->A, w->lda, ic, pc, bm, bk, dst);
                *shape = result;
                return 0;
            }
#endif
            int step =
                CAMBLAS_PACKED_DEEP_AMICRO32 >= 2 || shallow ? CAMBLAS_PACKED_A_COPY_DEPTH : bk;
            for (int j0 = 0; j0 < bk; j0 += step) {
                int end = bk - j0 < step ? bk : j0 + step;
                int i = 0;
                for (; i + 12 <= bm; i += 12)
                    for (int j = j0; j < end; j++)
                        memcpy(dst + (size_t)i * bk + (size_t)j * 12,
                               w->A + ic + i + (size_t)(pc + j) * w->lda, 12 * sizeof(float));
                if (i < bm)
                    for (int j = j0; j < end; j++) {
                        float *out = dst + (size_t)i * bk + (size_t)j * 12;
                        memcpy(out, w->A + ic + i + (size_t)(pc + j) * w->lda,
                               (size_t)(bm - i) * sizeof(float));
                        memset(out + bm - i, 0, (size_t)(12 - bm + i) * sizeof(float));
                    }
            }
            *shape = result;
            return 0;
        }
    }
    if (CAMBLAS_PACKED_DEEP_PAD_COPY && w->trans_a == 'N' && w->m > w->n && w->m <= 8192 &&
        w->n <= 2048 && w->k > 1024 && w->k <= 4096 && w->n_tasks >= 16) {
        camblas_packed_panel_t result;
        if (camblas_packed_panel_shape(w->m, w->k, ic, pc, bm, bk, alignment, 1, &result) ||
            result.elements > w->a_elements)
            return -1;
        for (int j = 0; j < bk; j++) {
            float *column = dst + (size_t)j * result.ld;
            memcpy(column, w->A + ic + (size_t)(pc + j) * w->lda, (size_t)bm * sizeof(float));
            memset(column + bm, 0, (result.padded_rows - (size_t)bm) * sizeof(float));
        }
        *shape = result;
        return 0;
    }
    return camblas_pack_f32_panel(w->trans_a, w->m, w->k, w->lda, ic, pc, bm, bk, alignment, 1,
                                  w->A, dst, w->a_elements, shape);
}
#define SHARED_A_PACK_F32(w, ic, pc, bm, bk, dst, shape) \
    sgemm_copy_shared_a(w, ic, pc, bm, bk, dst, shape)
static int dgemm_copy_shared_a(dgemm_packed_work_t *w, int ic, int pc, int bm, int bk, double *dst,
                               camblas_packed_panel_t *shape)
{
    if (CAMBLAS_PACKED_DEEP_AMICRO64 && w->nr == 8 && w->trans_a == 'N' && w->trans_b == 'N' &&
        w->m > w->n && w->m <= 8192 && w->n <= 2048 && w->k > 1024 && w->k <= 4096 &&
        w->n_tasks >= 16 && w->n_tasks <= CAMBLAS_PACKED_DEEP64_MAX_TASKS &&
        (!CAMBLAS_PACKED_DEEP64_MAX_ASPECT ||
         (uint64_t)w->m <= (uint64_t)CAMBLAS_PACKED_DEEP64_MAX_ASPECT * w->n)) {
        camblas_packed_panel_t result;
        if (camblas_packed_panel_shape(w->m, w->k, ic, pc, bm, bk, 6, 1, &result))
            return -1;
        /* Preserve the established allocation bound for irregular row blocks.
         * Grouping six rows together makes each inner-kernel A load contiguous. */
        if (result.elements <= w->a_elements) {
            result.layout = CAMBLAS_PANEL_A_MICRO;
            result.row_group = 6;
            result.ld = (size_t)bk;
            int step = CAMBLAS_PACKED_DEEP_AMICRO64 >= 2 ? CAMBLAS_PACKED_A_COPY_DEPTH : bk;
            for (int j0 = 0; j0 < bk; j0 += step) {
                int end = bk - j0 < step ? bk : j0 + step;
                int i = 0;
                for (; i + 6 <= bm; i += 6)
                    for (int j = j0; j < end; j++)
                        memcpy(dst + (size_t)i * bk + (size_t)j * 6,
                               w->A + ic + i + (size_t)(pc + j) * w->lda, 6 * sizeof(double));
                if (i < bm)
                    for (int j = j0; j < end; j++) {
                        double *out = dst + (size_t)i * bk + (size_t)j * 6;
                        memcpy(out, w->A + ic + i + (size_t)(pc + j) * w->lda,
                               (size_t)(bm - i) * sizeof(double));
                        memset(out + bm - i, 0, (size_t)(6 - bm + i) * sizeof(double));
                    }
            }
            *shape = result;
            return 0;
        }
    }
    return pack_a_f64(w->nr, w->trans_a, w->m, w->k, w->lda, ic, pc, bm, bk,
                      a_pack_alignment(w->mr, bm, sizeof(double)), 1, w->A, dst, w->a_elements,
                      shape);
}
#define SHARED_A_PACK_F64(w, ic, pc, bm, bk, dst, shape) \
    dgemm_copy_shared_a(w, ic, pc, bm, bk, dst, shape)
#define DEFINE_SHARED_A(SUFFIX, TYPE, WORK, PACK)                                               \
    static void SUFFIX##_shared_a_task(const camblas_task_t *task, void *opaque)                \
    {                                                                                           \
        WORK *w = opaque;                                                                       \
        for (int rb = task->i0; rb < task->i1; ++rb)                                            \
            for (int kb = task->j0; kb < task->j1; ++kb) {                                      \
                int ic = rb * w->mc, pc = kb * w->kc;                                           \
                int bm = w->m - ic < w->mc ? w->m - ic : w->mc,                                 \
                    bk = w->k - pc < w->kc ? w->k - pc : w->kc;                                 \
                size_t slot = (size_t)rb * w->shared_a_kblocks + kb;                            \
                camblas_packed_panel_t *shape = w->shared_a_shapes + slot;                      \
                if (PACK(w, ic, pc, bm, bk, w->shared_a + slot * w->a_elements, shape))         \
                    atomic_store_explicit(&w->failed, 1, memory_order_release);                 \
                else                                                                            \
                    packed_trace_note_pack(&w->trace, 1, shape->elements);                      \
            }                                                                                   \
    }                                                                                           \
    static int SUFFIX##_prepare_shared_a(WORK *w, const camblas_executor_t *executor)           \
    {                                                                                           \
        if (w->alpha == 0 || w->k == 0)                                                         \
            return 0;                                                                           \
        if (SUFFIX##_private_deep_a(w))                                                         \
            return 0;                                                                           \
        if (CAMBLAS_PACKED_ADAPTIVE_A && w->k <= 1024 && w->n <= 2048 &&                        \
            (CAMBLAS_PACKED_ADAPTIVE_A == 1 ||                                                  \
             (w->m <= 2048 &&                                                                   \
              (int64_t)ceil_div_positive(w->m, w->mc) * ceil_div_positive(w->n, w->nc) >= 32))) \
            return 0;                                                                           \
        int rb = ceil_div_positive(w->m, w->mc), kb = ceil_div_positive(w->k, w->kc);           \
        size_t slots, storage, descriptors;                                                     \
        if (checked_product((size_t)rb, (size_t)kb, &slots) ||                                  \
            checked_product(slots, w->a_bytes, &storage) ||                                     \
            checked_product(slots, sizeof(camblas_packed_panel_t), &descriptors) ||             \
            storage > PTRDIFF_MAX || descriptors > PTRDIFF_MAX)                                 \
            return -1;                                                                          \
        size_t b_elements, b_storage;                                                           \
        if (w->borrowed_b && w->shared_b == w->borrowed_b &&                                    \
            !checked_product(w->shared_npad, (size_t)w->k, &b_elements) &&                      \
            !checked_product(b_elements, sizeof(TYPE), &b_storage) &&                           \
            b_storage <= w->borrowed_bytes && storage <= w->borrowed_bytes - b_storage)         \
            w->shared_a = (TYPE *)((unsigned char *)w->borrowed_b + b_storage);                 \
        else {                                                                                  \
            w->shared_a = malloc(storage);                                                      \
            w->shared_a_owned = 1;                                                              \
        }                                                                                       \
        w->shared_a_shapes = SUFFIX##_shared_a_descriptors(w, storage, descriptors);            \
        w->shared_a_kblocks = kb;                                                               \
        if (!w->shared_a || !w->shared_a_shapes)                                                \
            return -1;                                                                          \
        camblas_task_t tasks[CAMBLAS_PACKED_MAX_TASKS];                                         \
        int count = split_packed_tasks(rb, kb, 1, 1, tasks, CAMBLAS_PACKED_MAX_TASKS);          \
        if (count < 1)                                                                          \
            return -1;                                                                          \
        if (SUFFIX##_defer_shared_a(w, tasks, count))                                           \
            return 0;                                                                           \
        int rc = executor->run(SUFFIX##_shared_a_task, tasks, count, w, executor->user_data);   \
        return rc || atomic_load_explicit(&w->failed, memory_order_acquire) ? -1 : 0;           \
    }
DEFINE_SHARED_A(sgemm, float, sgemm_packed_work_t, SHARED_A_PACK_F32)
DEFINE_SHARED_A(dgemm, double, dgemm_packed_work_t, SHARED_A_PACK_F64)
#undef DEFINE_SHARED_A
#undef SHARED_A_PACK_F32
#undef SHARED_A_PACK_F64
#endif

#if CAMBLAS_EXPERIMENTAL_SUM_PACK && CAMBLAS_SUM_PACK_ONE_PASS
static int sgemm_sum_pack_a(const sgemm_packed_work_t *w, int ic, int pc, int bm, int bk,
                            float *dst, camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result;
    if (camblas_packed_panel_shape(w->m, w->k, ic, pc, bm, bk,
                                   a_pack_alignment(w->mr, bm, sizeof(*dst)), 1, &result) ||
        result.elements > w->a_elements)
        return -1;
    for (int q = 0; q < bk; ++q) {
        const float *a = w->A + (size_t)(pc + q) * w->lda + ic;
        const float *b = w->A2 + (size_t)(pc + q) * w->lda2 + ic;
        float *out = dst + (size_t)q * result.ld;
        if (w->sign_a == 1)
            for (int i = 0; i < bm; ++i)
                out[i] = a[i] + b[i];
        else
            for (int i = 0; i < bm; ++i)
                out[i] = a[i] - b[i];
        for (size_t i = bm; i < result.ld; ++i)
            out[i] = 0;
    }
    *shape = result;
    return 0;
}
static int sgemm_sum_pack_b(const sgemm_packed_work_t *w, int pc, int jc, int bk, int bn,
                            float *dst, size_t capacity, camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result;
    if (camblas_packed_panel_shape(w->k, w->n, pc, jc, bk, bn, 1, 8, &result) ||
        result.elements > capacity)
        return -1;
    result.layout = CAMBLAS_PANEL_B_MICRO8;
    size_t j = 0;
    for (; j + 8 <= result.logical_cols; j += 8) {
        const float *a[8], *b[8];
        for (int z = 0; z < 8; ++z) {
            a[z] = w->B + (size_t)(jc + j + z) * w->ldb + pc;
            b[z] = w->B2 + (size_t)(jc + j + z) * w->ldb2 + pc;
        }
        float *out = dst + j * result.ld;
        int q = 0;
        /* Consume a full 64-byte source line before moving to other columns. */
        for (; q + 16 <= bk; q += 16)
            for (int group = 0; group < 8; group += 4) {
                float32x4_t v00 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 0] + q + 0),
                                                             vld1q_f32(b[group + 0] + q + 0))
                                                 : vsubq_f32(vld1q_f32(a[group + 0] + q + 0),
                                                             vld1q_f32(b[group + 0] + q + 0));
                float32x4_t v01 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 0] + q + 4),
                                                             vld1q_f32(b[group + 0] + q + 4))
                                                 : vsubq_f32(vld1q_f32(a[group + 0] + q + 4),
                                                             vld1q_f32(b[group + 0] + q + 4));
                float32x4_t v02 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 0] + q + 8),
                                                             vld1q_f32(b[group + 0] + q + 8))
                                                 : vsubq_f32(vld1q_f32(a[group + 0] + q + 8),
                                                             vld1q_f32(b[group + 0] + q + 8));
                float32x4_t v03 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 0] + q + 12),
                                                             vld1q_f32(b[group + 0] + q + 12))
                                                 : vsubq_f32(vld1q_f32(a[group + 0] + q + 12),
                                                             vld1q_f32(b[group + 0] + q + 12));
                float32x4_t v10 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 1] + q + 0),
                                                             vld1q_f32(b[group + 1] + q + 0))
                                                 : vsubq_f32(vld1q_f32(a[group + 1] + q + 0),
                                                             vld1q_f32(b[group + 1] + q + 0));
                float32x4_t v11 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 1] + q + 4),
                                                             vld1q_f32(b[group + 1] + q + 4))
                                                 : vsubq_f32(vld1q_f32(a[group + 1] + q + 4),
                                                             vld1q_f32(b[group + 1] + q + 4));
                float32x4_t v12 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 1] + q + 8),
                                                             vld1q_f32(b[group + 1] + q + 8))
                                                 : vsubq_f32(vld1q_f32(a[group + 1] + q + 8),
                                                             vld1q_f32(b[group + 1] + q + 8));
                float32x4_t v13 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 1] + q + 12),
                                                             vld1q_f32(b[group + 1] + q + 12))
                                                 : vsubq_f32(vld1q_f32(a[group + 1] + q + 12),
                                                             vld1q_f32(b[group + 1] + q + 12));
                float32x4_t v20 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 2] + q + 0),
                                                             vld1q_f32(b[group + 2] + q + 0))
                                                 : vsubq_f32(vld1q_f32(a[group + 2] + q + 0),
                                                             vld1q_f32(b[group + 2] + q + 0));
                float32x4_t v21 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 2] + q + 4),
                                                             vld1q_f32(b[group + 2] + q + 4))
                                                 : vsubq_f32(vld1q_f32(a[group + 2] + q + 4),
                                                             vld1q_f32(b[group + 2] + q + 4));
                float32x4_t v22 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 2] + q + 8),
                                                             vld1q_f32(b[group + 2] + q + 8))
                                                 : vsubq_f32(vld1q_f32(a[group + 2] + q + 8),
                                                             vld1q_f32(b[group + 2] + q + 8));
                float32x4_t v23 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 2] + q + 12),
                                                             vld1q_f32(b[group + 2] + q + 12))
                                                 : vsubq_f32(vld1q_f32(a[group + 2] + q + 12),
                                                             vld1q_f32(b[group + 2] + q + 12));
                float32x4_t v30 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 3] + q + 0),
                                                             vld1q_f32(b[group + 3] + q + 0))
                                                 : vsubq_f32(vld1q_f32(a[group + 3] + q + 0),
                                                             vld1q_f32(b[group + 3] + q + 0));
                float32x4_t v31 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 3] + q + 4),
                                                             vld1q_f32(b[group + 3] + q + 4))
                                                 : vsubq_f32(vld1q_f32(a[group + 3] + q + 4),
                                                             vld1q_f32(b[group + 3] + q + 4));
                float32x4_t v32 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 3] + q + 8),
                                                             vld1q_f32(b[group + 3] + q + 8))
                                                 : vsubq_f32(vld1q_f32(a[group + 3] + q + 8),
                                                             vld1q_f32(b[group + 3] + q + 8));
                float32x4_t v33 = w->sign_b == 1 ? vaddq_f32(vld1q_f32(a[group + 3] + q + 12),
                                                             vld1q_f32(b[group + 3] + q + 12))
                                                 : vsubq_f32(vld1q_f32(a[group + 3] + q + 12),
                                                             vld1q_f32(b[group + 3] + q + 12));
                float32x4_t t00 = vtrn1q_f32(v00, v10), t01 = vtrn2q_f32(v00, v10);
                float32x4_t t02 = vtrn1q_f32(v20, v30), t03 = vtrn2q_f32(v20, v30);
                vst1q_f32(out + (size_t)(q + 0) * 8 + group,
                          vcombine_f32(vget_low_f32(t00), vget_low_f32(t02)));
                vst1q_f32(out + (size_t)(q + 1) * 8 + group,
                          vcombine_f32(vget_low_f32(t01), vget_low_f32(t03)));
                vst1q_f32(out + (size_t)(q + 2) * 8 + group,
                          vcombine_f32(vget_high_f32(t00), vget_high_f32(t02)));
                vst1q_f32(out + (size_t)(q + 3) * 8 + group,
                          vcombine_f32(vget_high_f32(t01), vget_high_f32(t03)));
                float32x4_t t10 = vtrn1q_f32(v01, v11), t11 = vtrn2q_f32(v01, v11);
                float32x4_t t12 = vtrn1q_f32(v21, v31), t13 = vtrn2q_f32(v21, v31);
                vst1q_f32(out + (size_t)(q + 4) * 8 + group,
                          vcombine_f32(vget_low_f32(t10), vget_low_f32(t12)));
                vst1q_f32(out + (size_t)(q + 5) * 8 + group,
                          vcombine_f32(vget_low_f32(t11), vget_low_f32(t13)));
                vst1q_f32(out + (size_t)(q + 6) * 8 + group,
                          vcombine_f32(vget_high_f32(t10), vget_high_f32(t12)));
                vst1q_f32(out + (size_t)(q + 7) * 8 + group,
                          vcombine_f32(vget_high_f32(t11), vget_high_f32(t13)));
                float32x4_t t20 = vtrn1q_f32(v02, v12), t21 = vtrn2q_f32(v02, v12);
                float32x4_t t22 = vtrn1q_f32(v22, v32), t23 = vtrn2q_f32(v22, v32);
                vst1q_f32(out + (size_t)(q + 8) * 8 + group,
                          vcombine_f32(vget_low_f32(t20), vget_low_f32(t22)));
                vst1q_f32(out + (size_t)(q + 9) * 8 + group,
                          vcombine_f32(vget_low_f32(t21), vget_low_f32(t23)));
                vst1q_f32(out + (size_t)(q + 10) * 8 + group,
                          vcombine_f32(vget_high_f32(t20), vget_high_f32(t22)));
                vst1q_f32(out + (size_t)(q + 11) * 8 + group,
                          vcombine_f32(vget_high_f32(t21), vget_high_f32(t23)));
                float32x4_t t30 = vtrn1q_f32(v03, v13), t31 = vtrn2q_f32(v03, v13);
                float32x4_t t32 = vtrn1q_f32(v23, v33), t33 = vtrn2q_f32(v23, v33);
                vst1q_f32(out + (size_t)(q + 12) * 8 + group,
                          vcombine_f32(vget_low_f32(t30), vget_low_f32(t32)));
                vst1q_f32(out + (size_t)(q + 13) * 8 + group,
                          vcombine_f32(vget_low_f32(t31), vget_low_f32(t33)));
                vst1q_f32(out + (size_t)(q + 14) * 8 + group,
                          vcombine_f32(vget_high_f32(t30), vget_high_f32(t32)));
                vst1q_f32(out + (size_t)(q + 15) * 8 + group,
                          vcombine_f32(vget_high_f32(t31), vget_high_f32(t33)));
            }
        if (w->sign_b == 1)
            for (; q < bk; ++q) {
                out[(size_t)q * 8 + 0] = a[0][q] + b[0][q];
                out[(size_t)q * 8 + 1] = a[1][q] + b[1][q];
                out[(size_t)q * 8 + 2] = a[2][q] + b[2][q];
                out[(size_t)q * 8 + 3] = a[3][q] + b[3][q];
                out[(size_t)q * 8 + 4] = a[4][q] + b[4][q];
                out[(size_t)q * 8 + 5] = a[5][q] + b[5][q];
                out[(size_t)q * 8 + 6] = a[6][q] + b[6][q];
                out[(size_t)q * 8 + 7] = a[7][q] + b[7][q];
            }
        else
            for (; q < bk; ++q) {
                out[(size_t)q * 8 + 0] = a[0][q] - b[0][q];
                out[(size_t)q * 8 + 1] = a[1][q] - b[1][q];
                out[(size_t)q * 8 + 2] = a[2][q] - b[2][q];
                out[(size_t)q * 8 + 3] = a[3][q] - b[3][q];
                out[(size_t)q * 8 + 4] = a[4][q] - b[4][q];
                out[(size_t)q * 8 + 5] = a[5][q] - b[5][q];
                out[(size_t)q * 8 + 6] = a[6][q] - b[6][q];
                out[(size_t)q * 8 + 7] = a[7][q] - b[7][q];
            }
    }
    for (; j < result.padded_cols; j += 8)
        for (int q = 0; q < bk; ++q)
            for (int z = 0; z < 8; ++z) {
                float value = 0;
                if (j + z < result.logical_cols) {
                    float a = w->B[(size_t)(jc + j + z) * w->ldb + pc + q],
                          b = w->B2[(size_t)(jc + j + z) * w->ldb2 + pc + q];
                    value = w->sign_b == 1 ? a + b : a - b;
                }
                dst[j * result.ld + (size_t)q * 8 + z] = value;
            }
    *shape = result;
    return 0;
}
static int dgemm_sum_pack_a(const dgemm_packed_work_t *w, int ic, int pc, int bm, int bk,
                            double *dst, camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result;
    if (camblas_packed_panel_shape(w->m, w->k, ic, pc, bm, bk,
                                   a_pack_alignment(w->mr, bm, sizeof(*dst)), 1, &result) ||
        result.elements > w->a_elements)
        return -1;
    for (int q = 0; q < bk; ++q) {
        const double *a = w->A + (size_t)(pc + q) * w->lda + ic;
        const double *b = w->A2 + (size_t)(pc + q) * w->lda2 + ic;
        double *out = dst + (size_t)q * result.ld;
        if (w->sign_a == 1)
            for (int i = 0; i < bm; ++i)
                out[i] = a[i] + b[i];
        else
            for (int i = 0; i < bm; ++i)
                out[i] = a[i] - b[i];
        for (size_t i = bm; i < result.ld; ++i)
            out[i] = 0;
    }
    *shape = result;
    return 0;
}
static int dgemm_sum_pack_b(const dgemm_packed_work_t *w, int pc, int jc, int bk, int bn,
                            double *dst, size_t capacity, camblas_packed_panel_t *shape)
{
    camblas_packed_panel_t result;
    if (camblas_packed_panel_shape(w->k, w->n, pc, jc, bk, bn, 1, 8, &result) ||
        result.elements > capacity)
        return -1;
    result.layout = CAMBLAS_PANEL_B_MICRO8;
    size_t j = 0;
    for (; j + 8 <= result.logical_cols; j += 8) {
        const double *a[8], *b[8];
        for (int z = 0; z < 8; ++z) {
            a[z] = w->B + (size_t)(jc + j + z) * w->ldb + pc;
            b[z] = w->B2 + (size_t)(jc + j + z) * w->ldb2 + pc;
        }
        double *out = dst + j * result.ld;
        int q = 0;
        /* Consume a full 64-byte source line before moving to other columns. */
        for (; q + 8 <= bk; q += 8)
            for (int group = 0; group < 8; group += 4) {
                float64x2_t v00 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 0] + q + 0),
                                                             vld1q_f64(b[group + 0] + q + 0))
                                                 : vsubq_f64(vld1q_f64(a[group + 0] + q + 0),
                                                             vld1q_f64(b[group + 0] + q + 0));
                float64x2_t v01 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 0] + q + 2),
                                                             vld1q_f64(b[group + 0] + q + 2))
                                                 : vsubq_f64(vld1q_f64(a[group + 0] + q + 2),
                                                             vld1q_f64(b[group + 0] + q + 2));
                float64x2_t v02 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 0] + q + 4),
                                                             vld1q_f64(b[group + 0] + q + 4))
                                                 : vsubq_f64(vld1q_f64(a[group + 0] + q + 4),
                                                             vld1q_f64(b[group + 0] + q + 4));
                float64x2_t v03 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 0] + q + 6),
                                                             vld1q_f64(b[group + 0] + q + 6))
                                                 : vsubq_f64(vld1q_f64(a[group + 0] + q + 6),
                                                             vld1q_f64(b[group + 0] + q + 6));
                float64x2_t v10 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 1] + q + 0),
                                                             vld1q_f64(b[group + 1] + q + 0))
                                                 : vsubq_f64(vld1q_f64(a[group + 1] + q + 0),
                                                             vld1q_f64(b[group + 1] + q + 0));
                float64x2_t v11 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 1] + q + 2),
                                                             vld1q_f64(b[group + 1] + q + 2))
                                                 : vsubq_f64(vld1q_f64(a[group + 1] + q + 2),
                                                             vld1q_f64(b[group + 1] + q + 2));
                float64x2_t v12 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 1] + q + 4),
                                                             vld1q_f64(b[group + 1] + q + 4))
                                                 : vsubq_f64(vld1q_f64(a[group + 1] + q + 4),
                                                             vld1q_f64(b[group + 1] + q + 4));
                float64x2_t v13 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 1] + q + 6),
                                                             vld1q_f64(b[group + 1] + q + 6))
                                                 : vsubq_f64(vld1q_f64(a[group + 1] + q + 6),
                                                             vld1q_f64(b[group + 1] + q + 6));
                float64x2_t v20 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 2] + q + 0),
                                                             vld1q_f64(b[group + 2] + q + 0))
                                                 : vsubq_f64(vld1q_f64(a[group + 2] + q + 0),
                                                             vld1q_f64(b[group + 2] + q + 0));
                float64x2_t v21 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 2] + q + 2),
                                                             vld1q_f64(b[group + 2] + q + 2))
                                                 : vsubq_f64(vld1q_f64(a[group + 2] + q + 2),
                                                             vld1q_f64(b[group + 2] + q + 2));
                float64x2_t v22 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 2] + q + 4),
                                                             vld1q_f64(b[group + 2] + q + 4))
                                                 : vsubq_f64(vld1q_f64(a[group + 2] + q + 4),
                                                             vld1q_f64(b[group + 2] + q + 4));
                float64x2_t v23 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 2] + q + 6),
                                                             vld1q_f64(b[group + 2] + q + 6))
                                                 : vsubq_f64(vld1q_f64(a[group + 2] + q + 6),
                                                             vld1q_f64(b[group + 2] + q + 6));
                float64x2_t v30 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 3] + q + 0),
                                                             vld1q_f64(b[group + 3] + q + 0))
                                                 : vsubq_f64(vld1q_f64(a[group + 3] + q + 0),
                                                             vld1q_f64(b[group + 3] + q + 0));
                float64x2_t v31 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 3] + q + 2),
                                                             vld1q_f64(b[group + 3] + q + 2))
                                                 : vsubq_f64(vld1q_f64(a[group + 3] + q + 2),
                                                             vld1q_f64(b[group + 3] + q + 2));
                float64x2_t v32 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 3] + q + 4),
                                                             vld1q_f64(b[group + 3] + q + 4))
                                                 : vsubq_f64(vld1q_f64(a[group + 3] + q + 4),
                                                             vld1q_f64(b[group + 3] + q + 4));
                float64x2_t v33 = w->sign_b == 1 ? vaddq_f64(vld1q_f64(a[group + 3] + q + 6),
                                                             vld1q_f64(b[group + 3] + q + 6))
                                                 : vsubq_f64(vld1q_f64(a[group + 3] + q + 6),
                                                             vld1q_f64(b[group + 3] + q + 6));
                vst1q_f64(out + (size_t)(q + 0) * 8 + group + 0, vzip1q_f64(v00, v10));
                vst1q_f64(out + (size_t)(q + 0) * 8 + group + 2, vzip1q_f64(v20, v30));
                vst1q_f64(out + (size_t)(q + 1) * 8 + group + 0, vzip2q_f64(v00, v10));
                vst1q_f64(out + (size_t)(q + 1) * 8 + group + 2, vzip2q_f64(v20, v30));
                vst1q_f64(out + (size_t)(q + 2) * 8 + group + 0, vzip1q_f64(v01, v11));
                vst1q_f64(out + (size_t)(q + 2) * 8 + group + 2, vzip1q_f64(v21, v31));
                vst1q_f64(out + (size_t)(q + 3) * 8 + group + 0, vzip2q_f64(v01, v11));
                vst1q_f64(out + (size_t)(q + 3) * 8 + group + 2, vzip2q_f64(v21, v31));
                vst1q_f64(out + (size_t)(q + 4) * 8 + group + 0, vzip1q_f64(v02, v12));
                vst1q_f64(out + (size_t)(q + 4) * 8 + group + 2, vzip1q_f64(v22, v32));
                vst1q_f64(out + (size_t)(q + 5) * 8 + group + 0, vzip2q_f64(v02, v12));
                vst1q_f64(out + (size_t)(q + 5) * 8 + group + 2, vzip2q_f64(v22, v32));
                vst1q_f64(out + (size_t)(q + 6) * 8 + group + 0, vzip1q_f64(v03, v13));
                vst1q_f64(out + (size_t)(q + 6) * 8 + group + 2, vzip1q_f64(v23, v33));
                vst1q_f64(out + (size_t)(q + 7) * 8 + group + 0, vzip2q_f64(v03, v13));
                vst1q_f64(out + (size_t)(q + 7) * 8 + group + 2, vzip2q_f64(v23, v33));
            }
        if (w->sign_b == 1)
            for (; q < bk; ++q) {
                out[(size_t)q * 8 + 0] = a[0][q] + b[0][q];
                out[(size_t)q * 8 + 1] = a[1][q] + b[1][q];
                out[(size_t)q * 8 + 2] = a[2][q] + b[2][q];
                out[(size_t)q * 8 + 3] = a[3][q] + b[3][q];
                out[(size_t)q * 8 + 4] = a[4][q] + b[4][q];
                out[(size_t)q * 8 + 5] = a[5][q] + b[5][q];
                out[(size_t)q * 8 + 6] = a[6][q] + b[6][q];
                out[(size_t)q * 8 + 7] = a[7][q] + b[7][q];
            }
        else
            for (; q < bk; ++q) {
                out[(size_t)q * 8 + 0] = a[0][q] - b[0][q];
                out[(size_t)q * 8 + 1] = a[1][q] - b[1][q];
                out[(size_t)q * 8 + 2] = a[2][q] - b[2][q];
                out[(size_t)q * 8 + 3] = a[3][q] - b[3][q];
                out[(size_t)q * 8 + 4] = a[4][q] - b[4][q];
                out[(size_t)q * 8 + 5] = a[5][q] - b[5][q];
                out[(size_t)q * 8 + 6] = a[6][q] - b[6][q];
                out[(size_t)q * 8 + 7] = a[7][q] - b[7][q];
            }
    }
    for (; j < result.padded_cols; j += 8)
        for (int q = 0; q < bk; ++q)
            for (int z = 0; z < 8; ++z) {
                double value = 0;
                if (j + z < result.logical_cols) {
                    double a = w->B[(size_t)(jc + j + z) * w->ldb + pc + q],
                           b = w->B2[(size_t)(jc + j + z) * w->ldb2 + pc + q];
                    value = w->sign_b == 1 ? a + b : a - b;
                }
                dst[j * result.ld + (size_t)q * 8 + z] = value;
            }
    *shape = result;
    return 0;
}
#endif

static int packed_work_failed(const atomic_int *failed)
{
    return atomic_load_explicit(failed, memory_order_acquire) != 0;
}

#if defined(CAMBLAS_PACKED_SHARED_B) && CAMBLAS_PACKED_SHARED_B
#if !defined(CAMBLAS_PACKED_INTERLEAVED) || !CAMBLAS_PACKED_INTERLEAVED || \
    (defined(CAMBLAS_PACKED_MICRO6) && CAMBLAS_PACKED_MICRO6) ||           \
    (defined(CAMBLAS_PACKED_MICRO6_PAD8) && CAMBLAS_PACKED_MICRO6_PAD8)
#error "Shared B requires the micro8 packing contract"
#endif

#if CAMBLAS_PACKED_DEEP_VECTOR_B && defined(__aarch64__)
static void sgemm_transpose_four(const float *src, size_t stride, float *dst)
{
    float32x4_t a = vld1q_f32(src), b = vld1q_f32(src + stride);
    float32x4_t c = vld1q_f32(src + 2 * stride), d = vld1q_f32(src + 3 * stride);
    float64x2_t x = vreinterpretq_f64_f32(vtrn1q_f32(a, b));
    float64x2_t y = vreinterpretq_f64_f32(vtrn2q_f32(a, b));
    float64x2_t z = vreinterpretq_f64_f32(vtrn1q_f32(c, d));
    float64x2_t t = vreinterpretq_f64_f32(vtrn2q_f32(c, d));
    vst1q_f32(dst, vreinterpretq_f32_f64(vzip1q_f64(x, z)));
    vst1q_f32(dst + 8, vreinterpretq_f32_f64(vzip1q_f64(y, t)));
    vst1q_f32(dst + 16, vreinterpretq_f32_f64(vzip2q_f64(x, z)));
    vst1q_f32(dst + 24, vreinterpretq_f32_f64(vzip2q_f64(y, t)));
}
#endif
static int sgemm_pack_shared_b(sgemm_packed_work_t *w, int pc, int jc, int bk, int bn, float *dst,
                               size_t capacity, camblas_packed_panel_t *shape)
{
#if CAMBLAS_PACKED_DEEP_VECTOR_B && defined(__aarch64__)
    if (w->trans_b == 'N' && w->m > w->n && w->m <= 8192 && w->n <= 2048 && w->k > 1024 &&
        w->k <= 4096 && w->n_tasks >= CAMBLAS_PACKED_DEEP_MIN_TASKS && bk % 4 == 0 && bn % 8 == 0) {
        camblas_packed_panel_t result;
        if (camblas_packed_panel_shape(w->k, w->n, pc, jc, bk, bn, 1, 8, &result) ||
            result.elements > capacity)
            return -1;
        for (int j = 0; j < bn; j += 8)
            for (int q = 0; q < bk; q += 4) {
                const float *src = w->B + pc + q + (size_t)(jc + j) * w->ldb;
                float *out = dst + (size_t)j * bk + (size_t)q * 8;
                sgemm_transpose_four(src, (size_t)w->ldb, out);
                sgemm_transpose_four(src + (size_t)4 * w->ldb, (size_t)w->ldb, out + 4);
            }
        result.layout = CAMBLAS_PANEL_B_MICRO8;
        *shape = result;
        return 0;
    }
#endif
    return pack_b_f32(w->trans_b, w->k, w->n, w->ldb, pc, jc, bk, bn, 1, 8, w->B, dst, capacity,
                      shape);
}
static void sgemm_shared_b_task(const camblas_task_t *task, void *gctx)
{
    sgemm_packed_work_t *w = gctx;
    for (int pc = task->i0; pc < task->i1;) {
        int bk = w->k - pc < w->kc ? w->k - pc : w->kc;
        int bn = task->j1 - task->j0;
        size_t np = ((size_t)bn + 7) / 8 * 8;
        camblas_packed_panel_t shape;
        float *destination = shared_b_at(w->shared_b, w->shared_npad, pc, w->kc, sizeof(float));
        destination += (size_t)task->j0 * (size_t)bk;
        if (packed_work_failed(&w->failed))
            return;
#if CAMBLAS_EXPERIMENTAL_SUM_PACK && CAMBLAS_SUM_PACK_ONE_PASS
        if (w->B2) {
            if (sgemm_sum_pack_b(w, pc, task->j0, bk, bn, destination, np * (size_t)bk, &shape)) {
                atomic_store_explicit(&w->failed, 1, memory_order_release);
                return;
            }
        } else
#endif
            if (sgemm_pack_shared_b(w, pc, task->j0, bk, bn, destination, np * (size_t)bk,
                                    &shape) ||
                shape.layout != CAMBLAS_PANEL_B_MICRO8) {
            atomic_store_explicit(&w->failed, 1, memory_order_release);
            return;
        }
#if CAMBLAS_EXPERIMENTAL_SUM_PACK && !CAMBLAS_SUM_PACK_ONE_PASS
        if (w->B2)
            for (int j = 0; j < bn; ++j)
                for (int q = 0; q < bk; ++q) {
                    size_t dst = ((size_t)(j / 8) * bk + q) * 8 + j % 8;
                    float value = w->B2[(size_t)(task->j0 + j) * w->ldb2 + pc + q];
                    if (w->sign_b == 1)
                        destination[dst] += value;
                    else
                        destination[dst] -= value;
                }
#endif
        packed_trace_note_pack(&w->trace, 0, shape.elements);
        pc += bk;
    }
}

static int sgemm_prepare_shared_b(sgemm_packed_work_t *w, const camblas_executor_t *executor)
{
    if (!w->k || w->alpha == 0 || w->nr != 8 || w->nc % 8)
        return 0;
#if defined(CAMBLAS_PACKED_SHORT_PRIVATE) && CAMBLAS_PACKED_SHORT_PRIVATE
    /* An explicit workspace request retains the shared-B contract. */
    if (!w->borrowed_b && (w->k <= w->kc || w->k <= CAMBLAS_PACKED_PRIVATE_K_LIMIT))
        return 0;
#endif
    size_t elements, bytes;
    w->shared_npad = ((size_t)w->n + 7) / 8 * 8;
    if (checked_product(w->shared_npad, (size_t)w->k, &elements) ||
        checked_product(elements, sizeof(float), &bytes) || bytes > PTRDIFF_MAX)
        return -1;
    if (w->borrowed_b) {
        if (w->borrowed_bytes < bytes)
            return -1;
        w->shared_b = w->borrowed_b;
    } else
        w->shared_b = allocate_shared_b(w->shared_npad, w->k, w->kc, sizeof(float));
    if (!w->shared_b)
        return -1;
    camblas_task_t tasks[CAMBLAS_PACKED_MAX_TASKS];
    size_t width = ((size_t)ceil_div_positive(w->n, 256) + 7) / 8 * 8;
    size_t minimum = CAMBLAS_SHARED_B_MIN_WIDTH;
    if (CAMBLAS_SHARED_B_GUARD && (w->m > 2048 || w->n > 2048 || w->k > 1024 || w->n_tasks < 32))
        minimum = 64;
    if (CAMBLAS_SHARED_B_DEEP_WIDTH && w->m > w->n && w->m <= 8192 && w->n <= 2048 && w->k > 1024 &&
        w->k <= 4096 && w->n_tasks >= 16)
        minimum = CAMBLAS_SHARED_B_DEEP_WIDTH;
    if (width < minimum)
        width = minimum;
    int kstep = w->k;
    if (CAMBLAS_SHARED_B_DEEP_KBLOCKS > 0 && w->m > w->n && w->m <= 8192 && w->n <= 2048 &&
        w->k > 1024 && w->k <= 4096 && w->n_tasks >= 16) {
        int columns = ceil_div_positive(w->n, (int)width);
        int slots = CAMBLAS_PACKED_MAX_TASKS / columns;
        if (slots < 1)
            return -1;
        int group = ceil_div_positive(ceil_div_positive(w->k, w->kc), slots);
        if (group < CAMBLAS_SHARED_B_DEEP_KBLOCKS)
            group = CAMBLAS_SHARED_B_DEEP_KBLOCKS;
        int64_t span = (int64_t)group * w->kc;
        kstep = span < w->k ? (int)span : w->k;
    }
    int count = 0;
    for (size_t j = 0; j < (size_t)w->n; j += width) {
        size_t end = j + width < (size_t)w->n ? j + width : (size_t)w->n;
        for (int pc = 0; pc < w->k;) {
            int last = w->k - pc < kstep ? w->k : pc + kstep;
            if (count == CAMBLAS_PACKED_MAX_TASKS)
                return -1;
            tasks[count++] = (camblas_task_t){pc, last, (int)j, (int)end};
            pc = last;
        }
    }
#if CAMBLAS_PACKED_FUSED_PREP
    if (w->prepared_b) {
        for (int i = 0; i < count; i++)
            w->prepared_b[i] = tasks[i];
        w->prepared_b_count = count;
        return 0;
    }
#endif
    int rc = executor->run(sgemm_shared_b_task, tasks, count, w, executor->user_data);
    return rc || packed_work_failed(&w->failed) ? -1 : 0;
}

#if CAMBLAS_PACKED_FUSED_PREP && CAMBLAS_PACKED_SHARED_A
/* A and B own disjoint packing buffers. Spread each list over the same
 * bounded task index so a short B list does not occupy a separate team. */
static void sgemm_shared_ab_task(const camblas_task_t *task, void *gctx)
{
    sgemm_packed_work_t *w = gctx;
    int count =
        w->prepared_a_count > w->prepared_b_count ? w->prepared_a_count : w->prepared_b_count;
    int i = task->i0;
    int a0 = i * w->prepared_a_count / count, a1 = (i + 1) * w->prepared_a_count / count;
    int b0 = i * w->prepared_b_count / count, b1 = (i + 1) * w->prepared_b_count / count;
    if (a0 < a1)
        sgemm_shared_a_task(w->prepared_a + a0, w);
    if (b0 < b1)
        sgemm_shared_b_task(w->prepared_b + b0, w);
}
#endif

static void dgemm_shared_b_task(const camblas_task_t *task, void *gctx)
{
    dgemm_packed_work_t *w = gctx;
    for (int pc = 0; pc < w->k;) {
        int bk = w->k - pc < w->kc ? w->k - pc : w->kc;
        int bn = task->j1 - task->j0;
        size_t np = ((size_t)bn + 7) / 8 * 8;
        camblas_packed_panel_t shape;
        double *destination = shared_b_at(w->shared_b, w->shared_npad, pc, w->kc, sizeof(double));
        destination += (size_t)task->j0 * (size_t)bk;
        if (packed_work_failed(&w->failed))
            return;
#if CAMBLAS_EXPERIMENTAL_SUM_PACK && CAMBLAS_SUM_PACK_ONE_PASS
        if (w->B2) {
            if (dgemm_sum_pack_b(w, pc, task->j0, bk, bn, destination, np * (size_t)bk, &shape)) {
                atomic_store_explicit(&w->failed, 1, memory_order_release);
                return;
            }
        } else
#endif
            if (pack_b_f64(w->trans_b, w->k, w->n, w->ldb, pc, task->j0, bk, bn, 1, 8, w->B,
                           destination, np * (size_t)bk, &shape) ||
                shape.layout != CAMBLAS_PANEL_B_MICRO8) {
            atomic_store_explicit(&w->failed, 1, memory_order_release);
            return;
        }
#if CAMBLAS_EXPERIMENTAL_SUM_PACK && !CAMBLAS_SUM_PACK_ONE_PASS
        if (w->B2)
            for (int j = 0; j < bn; ++j)
                for (int q = 0; q < bk; ++q) {
                    size_t dst = ((size_t)(j / 8) * bk + q) * 8 + j % 8;
                    double value = w->B2[(size_t)(task->j0 + j) * w->ldb2 + pc + q];
                    if (w->sign_b == 1)
                        destination[dst] += value;
                    else
                        destination[dst] -= value;
                }
#endif
        packed_trace_note_pack(&w->trace, 0, shape.elements);
        pc += bk;
    }
}

static int dgemm_prepare_shared_b(dgemm_packed_work_t *w, const camblas_executor_t *executor)
{
    if (!w->k || w->alpha == 0 || w->nr != 8 || w->nc % 8)
        return 0;
#if defined(CAMBLAS_PACKED_SHORT_PRIVATE) && CAMBLAS_PACKED_SHORT_PRIVATE
    if (!w->borrowed_b && (w->k <= w->kc || w->k <= CAMBLAS_PACKED_PRIVATE_K_LIMIT))
        return 0;
#endif
    size_t elements, bytes;
    w->shared_npad = ((size_t)w->n + 7) / 8 * 8;
    if (checked_product(w->shared_npad, (size_t)w->k, &elements) ||
        checked_product(elements, sizeof(double), &bytes) || bytes > PTRDIFF_MAX)
        return -1;
    if (w->borrowed_b) {
        if (w->borrowed_bytes < bytes)
            return -1;
        w->shared_b = w->borrowed_b;
    } else
        w->shared_b = allocate_shared_b(w->shared_npad, w->k, w->kc, sizeof(double));
    if (!w->shared_b)
        return -1;
    camblas_task_t tasks[CAMBLAS_PACKED_MAX_TASKS];
    size_t width = ((size_t)ceil_div_positive(w->n, 256) + 7) / 8 * 8;
    size_t minimum = CAMBLAS_SHARED_B_MIN_WIDTH;
    if (CAMBLAS_SHARED_B_GUARD && (w->m > 2048 || w->n > 2048 || w->k > 1024 || w->n_tasks < 32))
        minimum = 64;
    if (width < minimum)
        width = minimum;
    int count = 0;
    for (size_t j = 0; j < (size_t)w->n; j += width) {
        size_t end = j + width < (size_t)w->n ? j + width : (size_t)w->n;
        if (count == CAMBLAS_PACKED_MAX_TASKS)
            return -1;
        tasks[count++] = (camblas_task_t){0, 1, (int)j, (int)end};
    }
    int rc = executor->run(dgemm_shared_b_task, tasks, count, w, executor->user_data);
    return rc || packed_work_failed(&w->failed) ? -1 : 0;
}

#endif

static void sgemm_packed_task_fn(const camblas_task_t *task, void *gctx)
{
    sgemm_packed_work_t *w = (sgemm_packed_work_t *)gctx;
    float *a_panel = NULL, *b_panel = NULL;

    if (!task || !w || packed_work_failed(&w->failed))
        return;
    /* Allocation is deliberately before C scaling.  A failed task allocation
       is post-dispatch and therefore reports EXECUTION_ERROR; a different
       task may already have changed its disjoint C rectangle. */
    if (w->alpha != 0.0f && w->k > 0) {
#if CAMBLAS_PACKED_SHARED_A
        if (!w->shared_a)
#endif
            a_panel = (float *)malloc(w->a_bytes);
        if (!w->shared_b)
            b_panel = (float *)malloc(w->b_bytes);
        if ((!a_panel
#if CAMBLAS_PACKED_SHARED_A
             && !w->shared_a
#endif
             ) ||
            (!w->shared_b && !b_panel))
            goto failed;
    }

    if (camblas_gemm_output_access_needed(w->k, w->alpha != 0.0f, w->beta != 1.0f) &&
        !packed_can_initialize(w->nr, w->alpha != 0 && w->k > 0, w->beta == 0)) {
        scale_f32(w->C + (size_t)task->i0 + (size_t)task->j0 * (size_t)w->ldc, task->i1 - task->i0,
                  task->j1 - task->j0, w->ldc, w->beta);
    }
    if (w->alpha == 0.0f || w->k == 0)
        goto done;

    for (int pc = 0; pc < w->k;) {
        int block_k = w->k - pc < w->kc ? w->k - pc : w->kc;
        int cached_b = CAMBLAS_PACKED_SUBTILES && !w->shared_b && task->j1 - task->j0 <= w->nc &&
                       task->i1 - task->i0 > w->mc;
        camblas_packed_panel_t cached_b_shape;
        if (cached_b) {
            if (pack_b_f32(w->trans_b, w->k, w->n, w->ldb, pc, task->j0, block_k,
                           task->j1 - task->j0, 1, w->nr, w->B, b_panel, w->b_elements,
                           &cached_b_shape))
                goto failed;
            packed_trace_note_pack(&w->trace, 0, cached_b_shape.elements);
        }
        for (int ic = task->i0; ic < task->i1;) {
            int block_m = task->i1 - ic < w->mc ? task->i1 - ic : w->mc;
            camblas_packed_panel_t a_shape;
            const float *tile_a = a_panel;
            if (packed_work_failed(&w->failed))
                goto done;
#if CAMBLAS_PACKED_SHARED_A
            if (w->shared_a) {
                size_t slot = (size_t)(ic / w->mc) * w->shared_a_kblocks + pc / w->kc;
                a_shape = w->shared_a_shapes[slot];
                tile_a = w->shared_a + slot * w->a_elements;
            } else
#endif
#if CAMBLAS_EXPERIMENTAL_SUM_PACK && CAMBLAS_SUM_PACK_ONE_PASS
                if (w->A2) {
                if (sgemm_sum_pack_a(w, ic, pc, block_m, block_k, a_panel, &a_shape))
                    goto failed;
            } else
#endif
#if CAMBLAS_PACKED_SHARED_A
                if (CAMBLAS_PACKED_SHALLOW_AMICRO32 && w->k <= 1024) {
                /* Private and shared panels follow the same checked policy.
                 * Unsupported shapes or capacities retain ordinary packing. */
                if (sgemm_copy_shared_a(w, ic, pc, block_m, block_k, a_panel, &a_shape))
                    goto failed;
            } else
#endif
                if (camblas_pack_f32_panel(w->trans_a, w->m, w->k, w->lda, ic, pc, block_m, block_k,
                                           a_pack_alignment(w->mr, block_m, sizeof(*a_panel)), 1,
                                           w->A, a_panel, w->a_elements, &a_shape) != 0)
                goto failed;
#if CAMBLAS_EXPERIMENTAL_SUM_PACK && !CAMBLAS_SUM_PACK_ONE_PASS
            if (w->A2)
                for (int q = 0; q < block_k; ++q)
                    for (int i = 0; i < block_m; ++i) {
                        size_t dst = (size_t)q * a_shape.ld + i;
                        float value = w->A2[(size_t)(pc + q) * w->lda2 + ic + i];
                        if (w->sign_a == 1)
                            a_panel[dst] += value;
                        else
                            a_panel[dst] -= value;
                    }
#endif
#if CAMBLAS_PACKED_SHARED_A
            if (!w->shared_a)
#endif
                packed_trace_note_pack(&w->trace, 1, a_shape.elements);
            for (int jc = task->j0; jc < task->j1;) {
                int block_n = task->j1 - jc < w->nc ? task->j1 - jc : w->nc;
                camblas_packed_panel_t b_shape;
                if (packed_work_failed(&w->failed))
                    goto done;
                const float *tile_b = b_panel;
                if (w->shared_b) {
                    size_t np = ((size_t)block_n + 7) / 8 * 8;
                    b_shape = (camblas_packed_panel_t){.logical_rows = (size_t)block_k,
                                                       .padded_rows = (size_t)block_k,
                                                       .logical_cols = (size_t)block_n,
                                                       .padded_cols = np,
                                                       .ld = (size_t)block_k,
                                                       .elements = np * (size_t)block_k,
                                                       .layout = CAMBLAS_PANEL_B_MICRO8};
                    tile_b = shared_b_at(w->shared_b, w->shared_npad, pc, w->kc, sizeof(float));
                    tile_b += (size_t)jc * (size_t)block_k;
                } else if (cached_b) {
                    b_shape = cached_b_shape;
                } else {
                    if (pack_b_f32(w->trans_b, w->k, w->n, w->ldb, pc, jc, block_k, block_n, 1,
                                   w->nr, w->B, b_panel, w->b_elements, &b_shape) != 0)
                        goto failed;
                    packed_trace_note_pack(&w->trace, 0, b_shape.elements);
                }
                if (compute_f32_tile(&a_shape, &b_shape, block_m, block_n, block_k, w->mr, w->nr,
                                     w->alpha, tile_a, tile_b, w->C, w->ldc, ic, jc, &w->trace,
                                     pc == 0 && packed_can_initialize(w->nr, 1, w->beta == 0)) != 0)
                    goto failed;
                jc += block_n;
            }
            ic += block_m;
        }
        pc += block_k;
    }
    goto done;

failed:
    atomic_store_explicit(&w->failed, 1, memory_order_release);
done:
    free(a_panel);
    free(b_panel);
}

static void dgemm_packed_task_fn(const camblas_task_t *task, void *gctx)
{
    dgemm_packed_work_t *w = (dgemm_packed_work_t *)gctx;
    double *a_panel = NULL, *b_panel = NULL;

    if (!task || !w || packed_work_failed(&w->failed))
        return;
    /* See the fp32 task: allocation precedes C scaling so this failure is
       never misreported as a retryable pre-dispatch UNAVAILABLE result. */
    if (w->alpha != 0.0 && w->k > 0) {
#if CAMBLAS_PACKED_SHARED_A
        if (!w->shared_a)
#endif
            a_panel = (double *)malloc(w->a_bytes);
        if (!w->shared_b)
            b_panel = (double *)malloc(w->b_bytes);
        if ((!a_panel
#if CAMBLAS_PACKED_SHARED_A
             && !w->shared_a
#endif
             ) ||
            (!w->shared_b && !b_panel))
            goto failed;
    }

    if (camblas_gemm_output_access_needed(w->k, w->alpha != 0.0, w->beta != 1.0) &&
        !packed_can_initialize(w->nr, w->alpha != 0 && w->k > 0, w->beta == 0)) {
        scale_f64(w->C + (size_t)task->i0 + (size_t)task->j0 * (size_t)w->ldc, task->i1 - task->i0,
                  task->j1 - task->j0, w->ldc, w->beta);
    }
    if (w->alpha == 0.0 || w->k == 0)
        goto done;

    for (int pc = 0; pc < w->k;) {
        int block_k = w->k - pc < w->kc ? w->k - pc : w->kc;
        int cached_b = CAMBLAS_PACKED_SUBTILES && !w->shared_b && task->j1 - task->j0 <= w->nc &&
                       task->i1 - task->i0 > w->mc;
        camblas_packed_panel_t cached_b_shape;
        if (cached_b) {
            if (pack_b_f64(w->trans_b, w->k, w->n, w->ldb, pc, task->j0, block_k,
                           task->j1 - task->j0, 1, w->nr, w->B, b_panel, w->b_elements,
                           &cached_b_shape))
                goto failed;
            packed_trace_note_pack(&w->trace, 0, cached_b_shape.elements);
        }
        for (int ic = task->i0; ic < task->i1;) {
            int block_m = task->i1 - ic < w->mc ? task->i1 - ic : w->mc;
            camblas_packed_panel_t a_shape;
            const double *tile_a = a_panel;
            if (packed_work_failed(&w->failed))
                goto done;
#if CAMBLAS_PACKED_SHARED_A
            if (w->shared_a) {
                size_t slot = (size_t)(ic / w->mc) * w->shared_a_kblocks + pc / w->kc;
                a_shape = w->shared_a_shapes[slot];
                tile_a = w->shared_a + slot * w->a_elements;
            } else
#endif
#if CAMBLAS_EXPERIMENTAL_SUM_PACK && CAMBLAS_SUM_PACK_ONE_PASS
                if (w->A2) {
                if (dgemm_sum_pack_a(w, ic, pc, block_m, block_k, a_panel, &a_shape))
                    goto failed;
            } else
#endif
                if (pack_a_f64(w->nr, w->trans_a, w->m, w->k, w->lda, ic, pc, block_m, block_k,
                               a_pack_alignment(w->mr, block_m, sizeof(*a_panel)), 1, w->A, a_panel,
                               w->a_elements, &a_shape) != 0)
                goto failed;
#if CAMBLAS_EXPERIMENTAL_SUM_PACK && !CAMBLAS_SUM_PACK_ONE_PASS
            if (w->A2)
                for (int q = 0; q < block_k; ++q)
                    for (int i = 0; i < block_m; ++i) {
                        size_t dst = (size_t)q * a_shape.ld + i;
                        double value = w->A2[(size_t)(pc + q) * w->lda2 + ic + i];
                        if (w->sign_a == 1)
                            a_panel[dst] += value;
                        else
                            a_panel[dst] -= value;
                    }
#endif
#if CAMBLAS_PACKED_SHARED_A
            if (!w->shared_a)
#endif
                packed_trace_note_pack(&w->trace, 1, a_shape.elements);
            for (int jc = task->j0; jc < task->j1;) {
                int block_n = task->j1 - jc < w->nc ? task->j1 - jc : w->nc;
                camblas_packed_panel_t b_shape;
                if (packed_work_failed(&w->failed))
                    goto done;
                const double *tile_b = b_panel;
                if (w->shared_b) {
                    size_t np = ((size_t)block_n + 7) / 8 * 8;
                    b_shape = (camblas_packed_panel_t){.logical_rows = (size_t)block_k,
                                                       .padded_rows = (size_t)block_k,
                                                       .logical_cols = (size_t)block_n,
                                                       .padded_cols = np,
                                                       .ld = (size_t)block_k,
                                                       .elements = np * (size_t)block_k,
                                                       .layout = CAMBLAS_PANEL_B_MICRO8};
                    tile_b = shared_b_at(w->shared_b, w->shared_npad, pc, w->kc, sizeof(double));
                    tile_b += (size_t)jc * (size_t)block_k;
                } else if (cached_b) {
                    b_shape = cached_b_shape;
                } else {
                    if (pack_b_f64(w->trans_b, w->k, w->n, w->ldb, pc, jc, block_k, block_n, 1,
                                   w->nr, w->B, b_panel, w->b_elements, &b_shape) != 0)
                        goto failed;
                    packed_trace_note_pack(&w->trace, 0, b_shape.elements);
                }
                if (compute_f64_tile(&a_shape, &b_shape, block_m, block_n, block_k, w->mr, w->nr,
                                     w->alpha, tile_a, tile_b, w->C, w->ldc, ic, jc, &w->trace,
                                     pc == 0 && packed_can_initialize(w->nr, 1, w->beta == 0)) != 0)
                    goto failed;
                jc += block_n;
            }
            ic += block_m;
        }
        pc += block_k;
    }
    goto done;

failed:
    atomic_store_explicit(&w->failed, 1, memory_order_release);
done:
    free(a_panel);
    free(b_panel);
}

#if CAMBLAS_EXPERIMENTAL_SUM_PACK
static int sgemm_sum_impl(char trans_a, char trans_b, int m, int n, int k, float alpha,
                          const float *A, int lda, const float *B, int ldb, float beta, float *C,
                          int ldc, int mc, int nc, int kc, int mr, int nr,
                          const camblas_executor_t *executor, void *workspace,
                          size_t workspace_bytes, const float *A2, int lda2, int sign_a,
                          const float *B2, int ldb2, int sign_b)
#else
int camblas_sgemm_packed_executor_workspace(char trans_a, char trans_b, int m, int n, int k,
                                            float alpha, const float *A, int lda, const float *B,
                                            int ldb, float beta, float *C, int ldc, int mc, int nc,
                                            int kc, int mr, int nr,
                                            const camblas_executor_t *executor, void *workspace,
                                            size_t workspace_bytes)
#endif
{
    size_t a_elements = 0, b_elements = 0, a_bytes = 0, b_bytes = 0;
    camblas_task_t tasks[CAMBLAS_PACKED_MAX_TASKS];
    sgemm_packed_work_t work;
    int n_tasks, run_rc, failed, product_needed;

#if CAMBLAS_EXPERIMENTAL_SUM_PACK
    if ((A2 && (trans_a != 'N' || lda2 < m || (sign_a != 1 && sign_a != -1))) ||
        (B2 && (trans_b != 'N' || ldb2 < k || (sign_b != 1 && sign_b != -1))) || (!A2 && sign_a) ||
        (!B2 && sign_b))
        return CAMBLAS_PACKED_UNAVAILABLE;
    if ((A2 || B2) && (!workspace || nr != 8 ||
                       camblas_gemm_access_spans_fit(trans_a, trans_b, m, n, k, A2 ? lda2 : lda,
                                                     B2 ? ldb2 : ldb, ldc, 1, 1, sizeof(float))))
        return CAMBLAS_PACKED_UNAVAILABLE;
#endif
    if (!executor || !executor->run ||
        valid_gemm_args(trans_a, trans_b, m, n, k, lda, ldb, ldc, A, B, C) != 0 ||
        camblas_gemm_access_spans_fit(
            trans_a, trans_b, m, n, k, lda, ldb, ldc, alpha != 0.0f && k > 0,
            camblas_gemm_output_access_needed(k, alpha != 0.0f, beta != 1.0f),
            sizeof(float)) != 0 ||
        mc < 1 || nc < 1 || kc < 1 || mr < 1 || nr < 1)
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (workspace && ((uintptr_t)workspace % _Alignof(double) || workspace_bytes > PTRDIFF_MAX))
        return CAMBLAS_PACKED_EXECUTION_ERROR;
    if (!workspace && workspace_bytes)
        return CAMBLAS_PACKED_EXECUTION_ERROR;
#if !defined(CAMBLAS_PACKED_SHARED_B) || !CAMBLAS_PACKED_SHARED_B || \
    (defined(CAMBLAS_PACKED_SHARED_B_CHUNKS) && CAMBLAS_PACKED_SHARED_B_CHUNKS)
    if (workspace)
        return CAMBLAS_PACKED_EXECUTION_ERROR;
#endif
    if (workspace && (nr != 8 || nc % 8))
        return CAMBLAS_PACKED_EXECUTION_ERROR;
    if (m == 0 || n == 0)
        return 0;
    if (!camblas_gemm_output_access_needed(k, alpha != 0.0f, beta != 1.0f))
        return 0;
    product_needed = alpha != 0.0f && k > 0;
    if (product_needed && (workspace_capacities(m, n, k, mc, nc, kc, mr, nr, sizeof(float),
                                                &a_elements, &b_elements) != 0 ||
                           checked_product(a_elements, sizeof(float), &a_bytes) != 0 ||
                           checked_product(b_elements, sizeof(float), &b_bytes) != 0))
        return CAMBLAS_PACKED_UNAVAILABLE;
    n_tasks = split_packed_tasks(m, n, mc, nc, tasks, CAMBLAS_PACKED_MAX_TASKS);
    if (n_tasks < 1)
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (CAMBLAS_PACKED_SUBTILES && (!workspace || CAMBLAS_PACKED_SUBTILES >= 3) && n_tasks >= 64 &&
        m <= 2048 && n <= 2048 && k <= 1024 && mc > 64 && mc % 64 == 0 && mr <= 64 &&
        64 % mr == 0) {
        mc = 64;
        int limit = CAMBLAS_PACKED_SUBTILES == 2 || CAMBLAS_PACKED_SUBTILES == 4 ? 256 : 128;
        if (kc > limit)
            kc = limit;
        if (workspace_capacities(m, n, k, mc, nc, kc, mr, nr, sizeof(float), &a_elements,
                                 &b_elements) ||
            checked_product(a_elements, sizeof(float), &a_bytes) ||
            checked_product(b_elements, sizeof(float), &b_bytes))
            return CAMBLAS_PACKED_UNAVAILABLE;
    }
    work = (sgemm_packed_work_t){
        .trans_a = trans_a,
        .trans_b = trans_b,
        .m = m,
        .n = n,
        .k = k,
        .lda = lda,
        .ldb = ldb,
        .ldc = ldc,
        .mc = mc,
        .nc = nc,
        .kc = kc,
        .mr = mr,
        .nr = nr,
        .n_tasks = n_tasks,
        .alpha = alpha,
        .beta = beta,
        .A = A,
        .B = B,
        .C = C,
#if CAMBLAS_EXPERIMENTAL_SUM_PACK
        .A2 = A2,
        .B2 = B2,
        .lda2 = lda2,
        .ldb2 = ldb2,
        .sign_a = sign_a,
        .sign_b = sign_b,
#endif
        .a_elements = a_elements,
        .b_elements = b_elements,
        .a_bytes = a_bytes,
        .b_bytes = b_bytes,
        .borrowed_b = workspace,
        .borrowed_bytes = workspace_bytes,
    };
    packed_trace_init(&work.trace);
    atomic_init(&work.failed, 0);
    run_rc = 0;
#if CAMBLAS_PACKED_FUSED_PREP && CAMBLAS_PACKED_SHARED_A && CAMBLAS_PACKED_SHARED_B
    camblas_task_t prepared_a[CAMBLAS_PACKED_MAX_TASKS], prepared_b[CAMBLAS_PACKED_MAX_TASKS];
    int fused = m > n && m <= 8192 && n <= 2048 && k > 1024 && k <= 4096 && n_tasks >= 16;
    if (fused) {
        work.prepared_a = prepared_a;
        work.prepared_b = prepared_b;
    }
#endif
#if defined(CAMBLAS_PACKED_SHARED_B) && CAMBLAS_PACKED_SHARED_B
    run_rc = sgemm_prepare_shared_b(&work, executor);
#endif
#if CAMBLAS_PACKED_SHARED_A
    if (!run_rc)
        run_rc = sgemm_prepare_shared_a(&work, executor);
#endif
#if CAMBLAS_PACKED_FUSED_PREP && CAMBLAS_PACKED_SHARED_A && CAMBLAS_PACKED_SHARED_B
    if (fused && !run_rc) {
        camblas_task_t pack_tasks[CAMBLAS_PACKED_MAX_TASKS];
        int count = work.prepared_a_count > work.prepared_b_count ? work.prepared_a_count
                                                                  : work.prepared_b_count;
        for (int i = 0; i < count; i++)
            pack_tasks[i] = (camblas_task_t){i, i + 1, 0, 1};
        if (count)
            run_rc =
                executor->run(sgemm_shared_ab_task, pack_tasks, count, &work, executor->user_data);
        if (packed_work_failed(&work.failed))
            run_rc = -1;
    }
#endif
    if (!run_rc)
        run_rc = executor->run(sgemm_packed_task_fn, tasks, n_tasks, &work, executor->user_data);
    if (!work.borrowed_b)
        release_shared_b(work.shared_b, k, kc);
#if CAMBLAS_PACKED_SHARED_A
    if (work.shared_a_owned)
        free(work.shared_a);
    sgemm_release_a_descriptors(&work);
#endif
    failed = packed_work_failed(&work.failed) || packed_trace_failed(&work.trace);
    packed_trace_report("fp32", "caller-executor", &work.trace);
    return (run_rc != 0 || failed) ? CAMBLAS_PACKED_EXECUTION_ERROR : 0;
}

#if CAMBLAS_EXPERIMENTAL_SUM_PACK
int camblas_sgemm_packed_executor_workspace(char trans_a, char trans_b, int m, int n, int k,
                                            float alpha, const float *A, int lda, const float *B,
                                            int ldb, float beta, float *C, int ldc, int mc, int nc,
                                            int kc, int mr, int nr,
                                            const camblas_executor_t *executor, void *workspace,
                                            size_t workspace_bytes)
{
    return sgemm_sum_impl(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, mc, nc,
                          kc, mr, nr, executor, workspace, workspace_bytes, NULL, 0, 0, NULL, 0, 0);
}
int camblas_experimental_sgemm_sum_workspace(char trans_a, char trans_b, int m, int n, int k,
                                             float alpha, const float *A, int lda, const float *B,
                                             int ldb, float beta, float *C, int ldc, int mc, int nc,
                                             int kc, int mr, int nr,
                                             const camblas_executor_t *executor, void *workspace,
                                             size_t workspace_bytes, const float *A2, int lda2,
                                             int sign_a, const float *B2, int ldb2, int sign_b)
{
    return sgemm_sum_impl(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, mc, nc,
                          kc, mr, nr, executor, workspace, workspace_bytes, A2, lda2, sign_a, B2,
                          ldb2, sign_b);
}
#endif

int camblas_sgemm_packed_executor(char trans_a, char trans_b, int m, int n, int k, float alpha,
                                  const float *A, int lda, const float *B, int ldb, float beta,
                                  float *C, int ldc, int mc, int nc, int kc, int mr, int nr,
                                  const camblas_executor_t *executor)
{
    return camblas_sgemm_packed_executor_workspace(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb,
                                                   beta, C, ldc, mc, nc, kc, mr, nr, executor, NULL,
                                                   0);
}

#if CAMBLAS_EXPERIMENTAL_SUM_PACK
static int dgemm_sum_impl(char trans_a, char trans_b, int m, int n, int k, double alpha,
                          const double *A, int lda, const double *B, int ldb, double beta,
                          double *C, int ldc, int mc, int nc, int kc, int mr, int nr,
                          const camblas_executor_t *executor, void *workspace,
                          size_t workspace_bytes, const double *A2, int lda2, int sign_a,
                          const double *B2, int ldb2, int sign_b)
#else
int camblas_dgemm_packed_executor_workspace(char trans_a, char trans_b, int m, int n, int k,
                                            double alpha, const double *A, int lda, const double *B,
                                            int ldb, double beta, double *C, int ldc, int mc,
                                            int nc, int kc, int mr, int nr,
                                            const camblas_executor_t *executor, void *workspace,
                                            size_t workspace_bytes)
#endif
{
    size_t a_elements = 0, b_elements = 0, a_bytes = 0, b_bytes = 0;
    camblas_task_t tasks[CAMBLAS_PACKED_MAX_TASKS];
    dgemm_packed_work_t work;
    int n_tasks, run_rc, failed, product_needed;

#if CAMBLAS_EXPERIMENTAL_SUM_PACK
    if ((A2 && (trans_a != 'N' || lda2 < m || (sign_a != 1 && sign_a != -1))) ||
        (B2 && (trans_b != 'N' || ldb2 < k || (sign_b != 1 && sign_b != -1))) || (!A2 && sign_a) ||
        (!B2 && sign_b))
        return CAMBLAS_PACKED_UNAVAILABLE;
    if ((A2 || B2) && (!workspace || nr != 8 ||
                       camblas_gemm_access_spans_fit(trans_a, trans_b, m, n, k, A2 ? lda2 : lda,
                                                     B2 ? ldb2 : ldb, ldc, 1, 1, sizeof(double))))
        return CAMBLAS_PACKED_UNAVAILABLE;
#endif
    if (!executor || !executor->run ||
        valid_gemm_args(trans_a, trans_b, m, n, k, lda, ldb, ldc, A, B, C) != 0 ||
        camblas_gemm_access_spans_fit(
            trans_a, trans_b, m, n, k, lda, ldb, ldc, alpha != 0.0 && k > 0,
            camblas_gemm_output_access_needed(k, alpha != 0.0, beta != 1.0), sizeof(double)) != 0 ||
        mc < 1 || nc < 1 || kc < 1 || mr < 1 || nr < 1)
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (workspace && ((uintptr_t)workspace % _Alignof(double) || workspace_bytes > PTRDIFF_MAX))
        return CAMBLAS_PACKED_EXECUTION_ERROR;
    if (!workspace && workspace_bytes)
        return CAMBLAS_PACKED_EXECUTION_ERROR;
#if !defined(CAMBLAS_PACKED_SHARED_B) || !CAMBLAS_PACKED_SHARED_B || \
    (defined(CAMBLAS_PACKED_SHARED_B_CHUNKS) && CAMBLAS_PACKED_SHARED_B_CHUNKS)
    if (workspace)
        return CAMBLAS_PACKED_EXECUTION_ERROR;
#endif
    if (workspace && (nr != 8 || nc % 8))
        return CAMBLAS_PACKED_EXECUTION_ERROR;
    if (m == 0 || n == 0)
        return 0;
    if (!camblas_gemm_output_access_needed(k, alpha != 0.0, beta != 1.0))
        return 0;
    product_needed = alpha != 0.0 && k > 0;
    if (product_needed && (workspace_capacities(m, n, k, mc, nc, kc, mr, nr, sizeof(double),
                                                &a_elements, &b_elements) != 0 ||
                           checked_product(a_elements, sizeof(double), &a_bytes) != 0 ||
                           checked_product(b_elements, sizeof(double), &b_bytes) != 0))
        return CAMBLAS_PACKED_UNAVAILABLE;
    n_tasks = split_packed_tasks(m, n, mc, nc, tasks, CAMBLAS_PACKED_MAX_TASKS);
    if (n_tasks < 1)
        return CAMBLAS_PACKED_UNAVAILABLE;
    if (CAMBLAS_PACKED_SUBTILES && (!workspace || CAMBLAS_PACKED_SUBTILES >= 3) && n_tasks >= 64 &&
        m <= 2048 && n <= 2048 && k <= 1024 && mc > 64 && mc % 64 == 0 && mr <= 64 &&
        64 % mr == 0) {
        mc = 64;
        int limit = CAMBLAS_PACKED_SUBTILES == 2 || CAMBLAS_PACKED_SUBTILES == 4 ? 128 : 64;
        if (kc > limit)
            kc = limit;
        if (workspace_capacities(m, n, k, mc, nc, kc, mr, nr, sizeof(double), &a_elements,
                                 &b_elements) ||
            checked_product(a_elements, sizeof(double), &a_bytes) ||
            checked_product(b_elements, sizeof(double), &b_bytes))
            return CAMBLAS_PACKED_UNAVAILABLE;
    }
    work = (dgemm_packed_work_t){
        .trans_a = trans_a,
        .trans_b = trans_b,
        .m = m,
        .n = n,
        .k = k,
        .lda = lda,
        .ldb = ldb,
        .ldc = ldc,
        .mc = mc,
        .nc = nc,
        .kc = kc,
        .mr = mr,
        .nr = nr,
        .n_tasks = n_tasks,
        .alpha = alpha,
        .beta = beta,
        .A = A,
        .B = B,
        .C = C,
#if CAMBLAS_EXPERIMENTAL_SUM_PACK
        .A2 = A2,
        .B2 = B2,
        .lda2 = lda2,
        .ldb2 = ldb2,
        .sign_a = sign_a,
        .sign_b = sign_b,
#endif
        .a_elements = a_elements,
        .b_elements = b_elements,
        .a_bytes = a_bytes,
        .b_bytes = b_bytes,
        .borrowed_b = workspace,
        .borrowed_bytes = workspace_bytes,
    };
    packed_trace_init(&work.trace);
    atomic_init(&work.failed, 0);
    run_rc = 0;
#if defined(CAMBLAS_PACKED_SHARED_B) && CAMBLAS_PACKED_SHARED_B
    run_rc = dgemm_prepare_shared_b(&work, executor);
#endif
#if CAMBLAS_PACKED_SHARED_A
    if (!run_rc)
        run_rc = dgemm_prepare_shared_a(&work, executor);
#endif
    if (!run_rc)
        run_rc = executor->run(dgemm_packed_task_fn, tasks, n_tasks, &work, executor->user_data);
    if (!work.borrowed_b)
        release_shared_b(work.shared_b, k, kc);
#if CAMBLAS_PACKED_SHARED_A
    if (work.shared_a_owned)
        free(work.shared_a);
    free(work.shared_a_shapes);
#endif
    failed = packed_work_failed(&work.failed) || packed_trace_failed(&work.trace);
    packed_trace_report("fp64", "caller-executor", &work.trace);
    return (run_rc != 0 || failed) ? CAMBLAS_PACKED_EXECUTION_ERROR : 0;
}

#if CAMBLAS_EXPERIMENTAL_SUM_PACK
int camblas_dgemm_packed_executor_workspace(char trans_a, char trans_b, int m, int n, int k,
                                            double alpha, const double *A, int lda, const double *B,
                                            int ldb, double beta, double *C, int ldc, int mc,
                                            int nc, int kc, int mr, int nr,
                                            const camblas_executor_t *executor, void *workspace,
                                            size_t workspace_bytes)
{
    return dgemm_sum_impl(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, mc, nc,
                          kc, mr, nr, executor, workspace, workspace_bytes, NULL, 0, 0, NULL, 0, 0);
}
int camblas_experimental_dgemm_sum_workspace(char trans_a, char trans_b, int m, int n, int k,
                                             double alpha, const double *A, int lda,
                                             const double *B, int ldb, double beta, double *C,
                                             int ldc, int mc, int nc, int kc, int mr, int nr,
                                             const camblas_executor_t *executor, void *workspace,
                                             size_t workspace_bytes, const double *A2, int lda2,
                                             int sign_a, const double *B2, int ldb2, int sign_b)
{
    return dgemm_sum_impl(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, mc, nc,
                          kc, mr, nr, executor, workspace, workspace_bytes, A2, lda2, sign_a, B2,
                          ldb2, sign_b);
}
#endif

int camblas_dgemm_packed_executor(char trans_a, char trans_b, int m, int n, int k, double alpha,
                                  const double *A, int lda, const double *B, int ldb, double beta,
                                  double *C, int ldc, int mc, int nc, int kc, int mr, int nr,
                                  const camblas_executor_t *executor)
{
    return camblas_dgemm_packed_executor_workspace(trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb,
                                                   beta, C, ldc, mc, nc, kc, mr, nr, executor, NULL,
                                                   0);
}

int camblas_sgemm_packed(char trans_a, char trans_b, int m, int n, int k, float alpha,
                         const float *A, int lda, const float *B, int ldb, float beta, float *C,
                         int ldc, int mc, int nc, int kc, int mr, int nr)
{
    size_t a_elements, b_elements, a_bytes, b_bytes;
    float *a_panel = NULL, *b_panel = NULL;
    packed_trace_t trace;

    if (valid_gemm_args(trans_a, trans_b, m, n, k, lda, ldb, ldc, A, B, C) != 0 ||
        camblas_gemm_access_spans_fit(
            trans_a, trans_b, m, n, k, lda, ldb, ldc, alpha != 0.0f && k > 0,
            camblas_gemm_output_access_needed(k, alpha != 0.0f, beta != 1.0f),
            sizeof(float)) != 0 ||
        mc < 1 || nc < 1 || kc < 1 || mr < 1 || nr < 1)
        return CAMBLAS_PACKED_UNAVAILABLE;
    packed_trace_init(&trace);
    if (m == 0 || n == 0)
        return 0;

    /* Preserve the no-product alpha=0/zero-K contract before touching A or B. */
    if (alpha == 0.0f || k == 0) {
        scale_f32(C, m, n, ldc, beta);
        return 0;
    }
    if (workspace_capacities(m, n, k, mc, nc, kc, mr, nr, sizeof(float), &a_elements,
                             &b_elements) != 0 ||
        checked_product(a_elements, sizeof(*a_panel), &a_bytes) != 0 ||
        checked_product(b_elements, sizeof(*b_panel), &b_bytes) != 0)
        return CAMBLAS_PACKED_UNAVAILABLE;

    a_panel = (float *)malloc(a_bytes);
    b_panel = (float *)malloc(b_bytes);
    if (!a_panel || !b_panel) {
        free(a_panel);
        free(b_panel);
        return CAMBLAS_PACKED_UNAVAILABLE;
    }

    /* No pack failure is expected after the complete argument/capacity check.
       If an invariant is violated, return a distinct error rather than
       silently falling back after C has been modified. */
    if (!packed_can_initialize(nr, 1, beta == 0))
        scale_f32(C, m, n, ldc, beta);
#if defined(CAMBLAS_PACKED_REUSE_B) && CAMBLAS_PACKED_REUSE_B
    /* Reuse B across row blocks; each output retains increasing PC order. */
    for (int jc = 0; jc < n;) {
        int block_n = n - jc < nc ? n - jc : nc;
        for (int pc = 0; pc < k;) {
            int block_k = k - pc < kc ? k - pc : kc;
            camblas_packed_panel_t b_shape;
            if (pack_b_f32(trans_b, k, n, ldb, pc, jc, block_k, block_n, 1, nr, B, b_panel,
                           b_elements, &b_shape) != 0) {
                free(a_panel);
                free(b_panel);
                return CAMBLAS_PACKED_EXECUTION_ERROR;
            }
            packed_trace_note_pack(&trace, 0, b_shape.elements);
            for (int ic = 0; ic < m;) {
                int block_m = m - ic < mc ? m - ic : mc;
                camblas_packed_panel_t a_shape;
                if (camblas_pack_f32_panel(trans_a, m, k, lda, ic, pc, block_m, block_k,
                                           a_pack_alignment(mr, block_m, sizeof(*a_panel)), 1, A,
                                           a_panel, a_elements, &a_shape) != 0) {
                    free(a_panel);
                    free(b_panel);
                    return CAMBLAS_PACKED_EXECUTION_ERROR;
                }
                packed_trace_note_pack(&trace, 1, a_shape.elements);
                if (compute_f32_tile(&a_shape, &b_shape, block_m, block_n, block_k, mr, nr, alpha,
                                     a_panel, b_panel, C, ldc, ic, jc, &trace,
                                     pc == 0 && packed_can_initialize(nr, 1, beta == 0)) != 0) {
                    free(a_panel);
                    free(b_panel);
                    return CAMBLAS_PACKED_EXECUTION_ERROR;
                }
                ic += block_m;
            }
            pc += block_k;
        }
        jc += block_n;
    }
#else
    for (int ic = 0; ic < m;) {
        int block_m = m - ic < mc ? m - ic : mc;
        for (int pc = 0; pc < k;) {
            int block_k = k - pc < kc ? k - pc : kc;
            camblas_packed_panel_t a_shape;
            if (camblas_pack_f32_panel(trans_a, m, k, lda, ic, pc, block_m, block_k,
                                       a_pack_alignment(mr, block_m, sizeof(*a_panel)), 1, A,
                                       a_panel, a_elements, &a_shape) != 0) {
                free(a_panel);
                free(b_panel);
                return CAMBLAS_PACKED_EXECUTION_ERROR;
            }
            packed_trace_note_pack(&trace, 1, a_shape.elements);
            for (int jc = 0; jc < n;) {
                int block_n = n - jc < nc ? n - jc : nc;
                camblas_packed_panel_t b_shape;
                if (pack_b_f32(trans_b, k, n, ldb, pc, jc, block_k, block_n, 1, nr, B, b_panel,
                               b_elements, &b_shape) != 0) {
                    free(a_panel);
                    free(b_panel);
                    return CAMBLAS_PACKED_EXECUTION_ERROR;
                }
                packed_trace_note_pack(&trace, 0, b_shape.elements);
                if (compute_f32_tile(&a_shape, &b_shape, block_m, block_n, block_k, mr, nr, alpha,
                                     a_panel, b_panel, C, ldc, ic, jc, &trace,
                                     pc == 0 && packed_can_initialize(nr, 1, beta == 0)) != 0) {
                    free(a_panel);
                    free(b_panel);
                    return CAMBLAS_PACKED_EXECUTION_ERROR;
                }
                if (n - jc <= nc)
                    break;
                jc += nc;
            }
            if (k - pc <= kc)
                break;
            pc += kc;
        }
        if (m - ic <= mc)
            break;
        ic += mc;
    }
#endif
    {
        int trace_failed = packed_trace_failed(&trace);
        packed_trace_report("fp32", "serial", &trace);
        if (trace_failed) {
            free(a_panel);
            free(b_panel);
            return CAMBLAS_PACKED_EXECUTION_ERROR;
        }
    }
    free(a_panel);
    free(b_panel);
    return 0;
}

int camblas_dgemm_packed(char trans_a, char trans_b, int m, int n, int k, double alpha,
                         const double *A, int lda, const double *B, int ldb, double beta, double *C,
                         int ldc, int mc, int nc, int kc, int mr, int nr)
{
    size_t a_elements, b_elements, a_bytes, b_bytes;
    double *a_panel = NULL, *b_panel = NULL;
    packed_trace_t trace;

    if (valid_gemm_args(trans_a, trans_b, m, n, k, lda, ldb, ldc, A, B, C) != 0 ||
        camblas_gemm_access_spans_fit(
            trans_a, trans_b, m, n, k, lda, ldb, ldc, alpha != 0.0 && k > 0,
            camblas_gemm_output_access_needed(k, alpha != 0.0, beta != 1.0), sizeof(double)) != 0 ||
        mc < 1 || nc < 1 || kc < 1 || mr < 1 || nr < 1)
        return CAMBLAS_PACKED_UNAVAILABLE;
    packed_trace_init(&trace);
    if (m == 0 || n == 0)
        return 0;
    if (alpha == 0.0 || k == 0) {
        scale_f64(C, m, n, ldc, beta);
        return 0;
    }
    if (workspace_capacities(m, n, k, mc, nc, kc, mr, nr, sizeof(double), &a_elements,
                             &b_elements) != 0 ||
        checked_product(a_elements, sizeof(*a_panel), &a_bytes) != 0 ||
        checked_product(b_elements, sizeof(*b_panel), &b_bytes) != 0)
        return CAMBLAS_PACKED_UNAVAILABLE;

    a_panel = (double *)malloc(a_bytes);
    b_panel = (double *)malloc(b_bytes);
    if (!a_panel || !b_panel) {
        free(a_panel);
        free(b_panel);
        return CAMBLAS_PACKED_UNAVAILABLE;
    }

    if (!packed_can_initialize(nr, 1, beta == 0))
        scale_f64(C, m, n, ldc, beta);
#if defined(CAMBLAS_PACKED_REUSE_B) && CAMBLAS_PACKED_REUSE_B
    /* Reuse B across row blocks; each output retains increasing PC order. */
    for (int jc = 0; jc < n;) {
        int block_n = n - jc < nc ? n - jc : nc;
        for (int pc = 0; pc < k;) {
            int block_k = k - pc < kc ? k - pc : kc;
            camblas_packed_panel_t b_shape;
            if (pack_b_f64(trans_b, k, n, ldb, pc, jc, block_k, block_n, 1, nr, B, b_panel,
                           b_elements, &b_shape) != 0) {
                free(a_panel);
                free(b_panel);
                return CAMBLAS_PACKED_EXECUTION_ERROR;
            }
            packed_trace_note_pack(&trace, 0, b_shape.elements);
            for (int ic = 0; ic < m;) {
                int block_m = m - ic < mc ? m - ic : mc;
                camblas_packed_panel_t a_shape;
                if (pack_a_f64(nr, trans_a, m, k, lda, ic, pc, block_m, block_k,
                               a_pack_alignment(mr, block_m, sizeof(*a_panel)), 1, A, a_panel,
                               a_elements, &a_shape) != 0) {
                    free(a_panel);
                    free(b_panel);
                    return CAMBLAS_PACKED_EXECUTION_ERROR;
                }
                packed_trace_note_pack(&trace, 1, a_shape.elements);
                if (compute_f64_tile(&a_shape, &b_shape, block_m, block_n, block_k, mr, nr, alpha,
                                     a_panel, b_panel, C, ldc, ic, jc, &trace,
                                     pc == 0 && packed_can_initialize(nr, 1, beta == 0)) != 0) {
                    free(a_panel);
                    free(b_panel);
                    return CAMBLAS_PACKED_EXECUTION_ERROR;
                }
                ic += block_m;
            }
            pc += block_k;
        }
        jc += block_n;
    }
#else
    for (int ic = 0; ic < m;) {
        int block_m = m - ic < mc ? m - ic : mc;
        for (int pc = 0; pc < k;) {
            int block_k = k - pc < kc ? k - pc : kc;
            camblas_packed_panel_t a_shape;
            if (pack_a_f64(nr, trans_a, m, k, lda, ic, pc, block_m, block_k,
                           a_pack_alignment(mr, block_m, sizeof(*a_panel)), 1, A, a_panel,
                           a_elements, &a_shape) != 0) {
                free(a_panel);
                free(b_panel);
                return CAMBLAS_PACKED_EXECUTION_ERROR;
            }
            packed_trace_note_pack(&trace, 1, a_shape.elements);
            for (int jc = 0; jc < n;) {
                int block_n = n - jc < nc ? n - jc : nc;
                camblas_packed_panel_t b_shape;
                if (pack_b_f64(trans_b, k, n, ldb, pc, jc, block_k, block_n, 1, nr, B, b_panel,
                               b_elements, &b_shape) != 0) {
                    free(a_panel);
                    free(b_panel);
                    return CAMBLAS_PACKED_EXECUTION_ERROR;
                }
                packed_trace_note_pack(&trace, 0, b_shape.elements);
                if (compute_f64_tile(&a_shape, &b_shape, block_m, block_n, block_k, mr, nr, alpha,
                                     a_panel, b_panel, C, ldc, ic, jc, &trace,
                                     pc == 0 && packed_can_initialize(nr, 1, beta == 0)) != 0) {
                    free(a_panel);
                    free(b_panel);
                    return CAMBLAS_PACKED_EXECUTION_ERROR;
                }
                if (n - jc <= nc)
                    break;
                jc += nc;
            }
            if (k - pc <= kc)
                break;
            pc += kc;
        }
        if (m - ic <= mc)
            break;
        ic += mc;
    }
#endif
    {
        int trace_failed = packed_trace_failed(&trace);
        packed_trace_report("fp64", "serial", &trace);
        if (trace_failed) {
            free(a_panel);
            free(b_panel);
            return CAMBLAS_PACKED_EXECUTION_ERROR;
        }
    }
    free(a_panel);
    free(b_panel);
    return 0;
}
