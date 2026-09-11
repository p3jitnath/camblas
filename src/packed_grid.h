#ifndef CAMBLAS_PACKED_GRID_H
#define CAMBLAS_PACKED_GRID_H
#include <stdint.h>
#ifndef CAMBLAS_PACKED_BALANCE_WAVES
#define CAMBLAS_PACKED_BALANCE_WAVES 0
#endif

/* Shrink blocks when the grid is underfilled or has large edge imbalance.
 * Search the executor's bounded 16-by-16 grid, keeping packing
 * alignment and the original capacity limits. No allocation or state. */
static inline void camblas_fill_packed_grid(int m, int n, int threads, int mr, int nr, int *mc,
                                            int *nc)
{
    if (m < 1 || n < 1 || threads < 2 || threads > 256 || mr < 1 || nr < 1 || !mc || !nc ||
        *mc < 1 || *nc < 1)
        return;
    int64_t rb = (int64_t)m / *mc + (m % *mc != 0);
    int64_t cb = (int64_t)n / *nc + (n % *nc != 0);
    /* Grouped grids are outside this heuristic. Compare actual area with
       equal largest-task rectangles. Each covered dimension is less than
       twice its positive int input, so their product fits uint64_t. */
    if (rb > 16 || cb > 16)
        return;
    uint64_t row_width = m < *mc ? m : *mc, col_width = n < *nc ? n : *nc;
    uint64_t covered = row_width * col_width * (uint64_t)rb * (uint64_t)cb;
    uint64_t actual = (uint64_t)m * (uint64_t)n;
    int slots = (int)((rb * cb + threads - 1) / threads) * threads;
    int wave_imbalance = CAMBLAS_PACKED_BALANCE_WAVES && slots - rb * cb > slots / 8;
    if (rb * cb >= threads && covered - actual <= covered / 8 && !wave_imbalance)
        return;
    int best_count = 257, best_m = *mc, best_n = *nc;
    int64_t best_area = INT64_MAX;
    uint64_t best_cost = UINT64_MAX;
    for (int rows = 1; rows <= 16; ++rows)
        for (int cols = 1; cols <= 16; ++cols) {
            int64_t cm = ((int64_t)m + rows - 1) / rows;
            int64_t cn = ((int64_t)n + cols - 1) / cols;
            cm = (cm + mr - 1) / mr * mr;
            cn = (cn + nr - 1) / nr * nr;
            if (cm > *mc || cn > *nc)
                continue;
            int actual_r = (int)(((int64_t)m + cm - 1) / cm);
            int actual_c = (int)(((int64_t)n + cn - 1) / cn);
            int count = actual_r * actual_c;
            if (count < threads || actual_r > 16 || actual_c > 16)
                continue;
            int64_t area = cm * cn;
            /* Waves times largest task area estimates the compute critical path.
           This product is bounded by the grid's covered area and fits u64. */
            uint64_t cost = (uint64_t)area * (uint64_t)((count + threads - 1) / threads);
            int better_count =
                count < best_count ||
                (count == best_count && (area < best_area || (area == best_area && cn > best_n)));
            int better = CAMBLAS_PACKED_BALANCE_WAVES
                             ? (cost < best_cost || (cost == best_cost && better_count))
                             : better_count;
            if (better) {
                best_count = count;
                best_area = area;
                best_m = (int)cm;
                best_n = (int)cn;
                best_cost = cost;
            }
        }
    *mc = best_m;
    *nc = best_n;
}
#endif
