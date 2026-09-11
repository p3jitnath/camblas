#ifndef CAMBLAS_SVE_LANE_OPS_H
#define CAMBLAS_SVE_LANE_OPS_H
#ifndef CAMBLAS_SVE_ORDERED_LANE
#define CAMBLAS_SVE_ORDERED_LANE 0
#endif
#if CAMBLAS_SVE_ORDERED_LANE
/* Volatile indexed FMAs preserve the independent-chain instruction order.
 * y constrains indexed source operands to the encodable low eight registers. */
#define CAMBLAS_LANE_F32(acc, a, b, lane) \
    __asm__ volatile("fmla %0.s, %1.s, %2.s[%3]" : "+w"(acc) : "w"(a), "y"(b), "i"(lane))
#define CAMBLAS_LANE_F64(acc, a, b, lane) \
    __asm__ volatile("fmla %0.d, %1.d, %2.d[%3]" : "+w"(acc) : "w"(a), "y"(b), "i"(lane))
#else
#define CAMBLAS_LANE_F32(acc, a, b, lane) ((acc) = svmla_lane_f32((acc), (a), (b), (lane)))
#define CAMBLAS_LANE_F64(acc, a, b, lane) ((acc) = svmla_lane_f64((acc), (a), (b), (lane)))
#endif
#ifndef CAMBLAS_SVE_LANE_FMA
#define CAMBLAS_SVE_LANE_FMA 0
#endif
#endif
