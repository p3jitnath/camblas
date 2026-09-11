/* Beta-zero output preparation. Each virtual page receives one byte store,
 * owned by the first logical column intersecting that page. No C value is
 * read, no padding is touched, and GEMM subsequently overwrites every byte
 * of each logical output element. This is not result caching. */
#ifndef CAMBLAS_OUTPUT_PAGE_TOUCH_H
#define CAMBLAS_OUTPUT_PAGE_TOUCH_H
#include <stdint.h>
#include <stddef.h>
typedef struct {
    void *c;
    int m, n, ldc;
    size_t element_bytes, page_bytes;
    int triangle; /* 0 full, 1 upper, 2 lower; triangular outputs are square. */
} framework_output_touch_t;
static inline void framework_touch_column(const framework_output_touch_t *v, int j)
{
    uintptr_t column = (uintptr_t)v->c + (size_t)j * v->ldc * v->element_bytes;
    uintptr_t base = column + (v->triangle == 2 ? (size_t)j * v->element_bytes : 0);
    uintptr_t end = column + (size_t)(v->triangle == 1 ? j + 1 : v->m) * v->element_bytes;
    uintptr_t mask = v->page_bytes - 1;
    uintptr_t previous_end = j ? column - (size_t)v->ldc * v->element_bytes +
                                     (size_t)(v->triangle == 1 ? j : v->m) * v->element_bytes - 1
                               : 0;
    if (!j || (base & ~mask) > (previous_end & ~mask))
        *(volatile unsigned char *)base = 0;
    for (uintptr_t at = (base | mask) + 1; at < end; at += v->page_bytes)
        *(volatile unsigned char *)at = 0;
}
#endif
