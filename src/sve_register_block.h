/* Generated explicit 6-vector x 4-column SVE block. Full tiles only.
 * Callers retain the existing predicated edge path. Each output accumulator
 * follows the original K order; no reassociation or split reduction. */
#ifndef CAMBLAS_SVE_REGISTER_BLOCK_H
#define CAMBLAS_SVE_REGISTER_BLOCK_H
static inline __attribute__((always_inline)) void camblas_block6_f32(int k, float alpha,
                                                                     const float *A, size_t lda,
                                                                     const float *B, size_t ldb,
                                                                     float *C, size_t ldc)
{
    const size_t lanes = svcntw();
    const svbool_t pg = svptrue_b32();
    svfloat32_t acc00 = svdup_f32(0.0);
    svfloat32_t acc01 = svdup_f32(0.0);
    svfloat32_t acc02 = svdup_f32(0.0);
    svfloat32_t acc03 = svdup_f32(0.0);
    svfloat32_t acc10 = svdup_f32(0.0);
    svfloat32_t acc11 = svdup_f32(0.0);
    svfloat32_t acc12 = svdup_f32(0.0);
    svfloat32_t acc13 = svdup_f32(0.0);
    svfloat32_t acc20 = svdup_f32(0.0);
    svfloat32_t acc21 = svdup_f32(0.0);
    svfloat32_t acc22 = svdup_f32(0.0);
    svfloat32_t acc23 = svdup_f32(0.0);
    svfloat32_t acc30 = svdup_f32(0.0);
    svfloat32_t acc31 = svdup_f32(0.0);
    svfloat32_t acc32 = svdup_f32(0.0);
    svfloat32_t acc33 = svdup_f32(0.0);
    svfloat32_t acc40 = svdup_f32(0.0);
    svfloat32_t acc41 = svdup_f32(0.0);
    svfloat32_t acc42 = svdup_f32(0.0);
    svfloat32_t acc43 = svdup_f32(0.0);
    svfloat32_t acc50 = svdup_f32(0.0);
    svfloat32_t acc51 = svdup_f32(0.0);
    svfloat32_t acc52 = svdup_f32(0.0);
    svfloat32_t acc53 = svdup_f32(0.0);
    for (int l = 0; l < k; ++l) {
        svfloat32_t b0 = camblas_sve_load_replicate_f32(pg, B + (size_t)l + 0 * ldb);
        svfloat32_t b1 = camblas_sve_load_replicate_f32(pg, B + (size_t)l + 1 * ldb);
        svfloat32_t b2 = camblas_sve_load_replicate_f32(pg, B + (size_t)l + 2 * ldb);
        svfloat32_t b3 = camblas_sve_load_replicate_f32(pg, B + (size_t)l + 3 * ldb);
        svfloat32_t a0 = svld1_f32(pg, A + (size_t)l * lda + 0 * lanes);
        acc00 = svmla_f32_x(pg, acc00, a0, b0);
        acc01 = svmla_f32_x(pg, acc01, a0, b1);
        acc02 = svmla_f32_x(pg, acc02, a0, b2);
        acc03 = svmla_f32_x(pg, acc03, a0, b3);
        svfloat32_t a1 = svld1_f32(pg, A + (size_t)l * lda + 1 * lanes);
        acc10 = svmla_f32_x(pg, acc10, a1, b0);
        acc11 = svmla_f32_x(pg, acc11, a1, b1);
        acc12 = svmla_f32_x(pg, acc12, a1, b2);
        acc13 = svmla_f32_x(pg, acc13, a1, b3);
        svfloat32_t a2 = svld1_f32(pg, A + (size_t)l * lda + 2 * lanes);
        acc20 = svmla_f32_x(pg, acc20, a2, b0);
        acc21 = svmla_f32_x(pg, acc21, a2, b1);
        acc22 = svmla_f32_x(pg, acc22, a2, b2);
        acc23 = svmla_f32_x(pg, acc23, a2, b3);
        svfloat32_t a3 = svld1_f32(pg, A + (size_t)l * lda + 3 * lanes);
        acc30 = svmla_f32_x(pg, acc30, a3, b0);
        acc31 = svmla_f32_x(pg, acc31, a3, b1);
        acc32 = svmla_f32_x(pg, acc32, a3, b2);
        acc33 = svmla_f32_x(pg, acc33, a3, b3);
        svfloat32_t a4 = svld1_f32(pg, A + (size_t)l * lda + 4 * lanes);
        acc40 = svmla_f32_x(pg, acc40, a4, b0);
        acc41 = svmla_f32_x(pg, acc41, a4, b1);
        acc42 = svmla_f32_x(pg, acc42, a4, b2);
        acc43 = svmla_f32_x(pg, acc43, a4, b3);
        svfloat32_t a5 = svld1_f32(pg, A + (size_t)l * lda + 5 * lanes);
        acc50 = svmla_f32_x(pg, acc50, a5, b0);
        acc51 = svmla_f32_x(pg, acc51, a5, b1);
        acc52 = svmla_f32_x(pg, acc52, a5, b2);
        acc53 = svmla_f32_x(pg, acc53, a5, b3);
    }
    {
        float *out = C + 0 * lanes + 0 * ldc;
        svfloat32_t value = acc00;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 0 * lanes + 1 * ldc;
        svfloat32_t value = acc01;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 0 * lanes + 2 * ldc;
        svfloat32_t value = acc02;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 0 * lanes + 3 * ldc;
        svfloat32_t value = acc03;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 1 * lanes + 0 * ldc;
        svfloat32_t value = acc10;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 1 * lanes + 1 * ldc;
        svfloat32_t value = acc11;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 1 * lanes + 2 * ldc;
        svfloat32_t value = acc12;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 1 * lanes + 3 * ldc;
        svfloat32_t value = acc13;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 2 * lanes + 0 * ldc;
        svfloat32_t value = acc20;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 2 * lanes + 1 * ldc;
        svfloat32_t value = acc21;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 2 * lanes + 2 * ldc;
        svfloat32_t value = acc22;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 2 * lanes + 3 * ldc;
        svfloat32_t value = acc23;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 3 * lanes + 0 * ldc;
        svfloat32_t value = acc30;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 3 * lanes + 1 * ldc;
        svfloat32_t value = acc31;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 3 * lanes + 2 * ldc;
        svfloat32_t value = acc32;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 3 * lanes + 3 * ldc;
        svfloat32_t value = acc33;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 4 * lanes + 0 * ldc;
        svfloat32_t value = acc40;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 4 * lanes + 1 * ldc;
        svfloat32_t value = acc41;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 4 * lanes + 2 * ldc;
        svfloat32_t value = acc42;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 4 * lanes + 3 * ldc;
        svfloat32_t value = acc43;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 5 * lanes + 0 * ldc;
        svfloat32_t value = acc50;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 5 * lanes + 1 * ldc;
        svfloat32_t value = acc51;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 5 * lanes + 2 * ldc;
        svfloat32_t value = acc52;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
    {
        float *out = C + 5 * lanes + 3 * ldc;
        svfloat32_t value = acc53;
        if (alpha != 1.0)
            value = svmul_n_f32_x(pg, value, alpha);
        svst1_f32(pg, out, svadd_f32_x(pg, svld1_f32(pg, out), value));
    }
}
static inline __attribute__((always_inline)) void camblas_block6_f64(int k, double alpha,
                                                                     const double *A, size_t lda,
                                                                     const double *B, size_t ldb,
                                                                     double *C, size_t ldc)
{
    const size_t lanes = svcntd();
    const svbool_t pg = svptrue_b64();
    svfloat64_t acc00 = svdup_f64(0.0);
    svfloat64_t acc01 = svdup_f64(0.0);
    svfloat64_t acc02 = svdup_f64(0.0);
    svfloat64_t acc03 = svdup_f64(0.0);
    svfloat64_t acc10 = svdup_f64(0.0);
    svfloat64_t acc11 = svdup_f64(0.0);
    svfloat64_t acc12 = svdup_f64(0.0);
    svfloat64_t acc13 = svdup_f64(0.0);
    svfloat64_t acc20 = svdup_f64(0.0);
    svfloat64_t acc21 = svdup_f64(0.0);
    svfloat64_t acc22 = svdup_f64(0.0);
    svfloat64_t acc23 = svdup_f64(0.0);
    svfloat64_t acc30 = svdup_f64(0.0);
    svfloat64_t acc31 = svdup_f64(0.0);
    svfloat64_t acc32 = svdup_f64(0.0);
    svfloat64_t acc33 = svdup_f64(0.0);
    svfloat64_t acc40 = svdup_f64(0.0);
    svfloat64_t acc41 = svdup_f64(0.0);
    svfloat64_t acc42 = svdup_f64(0.0);
    svfloat64_t acc43 = svdup_f64(0.0);
    svfloat64_t acc50 = svdup_f64(0.0);
    svfloat64_t acc51 = svdup_f64(0.0);
    svfloat64_t acc52 = svdup_f64(0.0);
    svfloat64_t acc53 = svdup_f64(0.0);
    for (int l = 0; l < k; ++l) {
        svfloat64_t b0 = camblas_sve_load_replicate_f64(pg, B + (size_t)l + 0 * ldb);
        svfloat64_t b1 = camblas_sve_load_replicate_f64(pg, B + (size_t)l + 1 * ldb);
        svfloat64_t b2 = camblas_sve_load_replicate_f64(pg, B + (size_t)l + 2 * ldb);
        svfloat64_t b3 = camblas_sve_load_replicate_f64(pg, B + (size_t)l + 3 * ldb);
        svfloat64_t a0 = svld1_f64(pg, A + (size_t)l * lda + 0 * lanes);
        acc00 = svmla_f64_x(pg, acc00, a0, b0);
        acc01 = svmla_f64_x(pg, acc01, a0, b1);
        acc02 = svmla_f64_x(pg, acc02, a0, b2);
        acc03 = svmla_f64_x(pg, acc03, a0, b3);
        svfloat64_t a1 = svld1_f64(pg, A + (size_t)l * lda + 1 * lanes);
        acc10 = svmla_f64_x(pg, acc10, a1, b0);
        acc11 = svmla_f64_x(pg, acc11, a1, b1);
        acc12 = svmla_f64_x(pg, acc12, a1, b2);
        acc13 = svmla_f64_x(pg, acc13, a1, b3);
        svfloat64_t a2 = svld1_f64(pg, A + (size_t)l * lda + 2 * lanes);
        acc20 = svmla_f64_x(pg, acc20, a2, b0);
        acc21 = svmla_f64_x(pg, acc21, a2, b1);
        acc22 = svmla_f64_x(pg, acc22, a2, b2);
        acc23 = svmla_f64_x(pg, acc23, a2, b3);
        svfloat64_t a3 = svld1_f64(pg, A + (size_t)l * lda + 3 * lanes);
        acc30 = svmla_f64_x(pg, acc30, a3, b0);
        acc31 = svmla_f64_x(pg, acc31, a3, b1);
        acc32 = svmla_f64_x(pg, acc32, a3, b2);
        acc33 = svmla_f64_x(pg, acc33, a3, b3);
        svfloat64_t a4 = svld1_f64(pg, A + (size_t)l * lda + 4 * lanes);
        acc40 = svmla_f64_x(pg, acc40, a4, b0);
        acc41 = svmla_f64_x(pg, acc41, a4, b1);
        acc42 = svmla_f64_x(pg, acc42, a4, b2);
        acc43 = svmla_f64_x(pg, acc43, a4, b3);
        svfloat64_t a5 = svld1_f64(pg, A + (size_t)l * lda + 5 * lanes);
        acc50 = svmla_f64_x(pg, acc50, a5, b0);
        acc51 = svmla_f64_x(pg, acc51, a5, b1);
        acc52 = svmla_f64_x(pg, acc52, a5, b2);
        acc53 = svmla_f64_x(pg, acc53, a5, b3);
    }
    {
        double *out = C + 0 * lanes + 0 * ldc;
        svfloat64_t value = acc00;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 0 * lanes + 1 * ldc;
        svfloat64_t value = acc01;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 0 * lanes + 2 * ldc;
        svfloat64_t value = acc02;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 0 * lanes + 3 * ldc;
        svfloat64_t value = acc03;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 1 * lanes + 0 * ldc;
        svfloat64_t value = acc10;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 1 * lanes + 1 * ldc;
        svfloat64_t value = acc11;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 1 * lanes + 2 * ldc;
        svfloat64_t value = acc12;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 1 * lanes + 3 * ldc;
        svfloat64_t value = acc13;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 2 * lanes + 0 * ldc;
        svfloat64_t value = acc20;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 2 * lanes + 1 * ldc;
        svfloat64_t value = acc21;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 2 * lanes + 2 * ldc;
        svfloat64_t value = acc22;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 2 * lanes + 3 * ldc;
        svfloat64_t value = acc23;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 3 * lanes + 0 * ldc;
        svfloat64_t value = acc30;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 3 * lanes + 1 * ldc;
        svfloat64_t value = acc31;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 3 * lanes + 2 * ldc;
        svfloat64_t value = acc32;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 3 * lanes + 3 * ldc;
        svfloat64_t value = acc33;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 4 * lanes + 0 * ldc;
        svfloat64_t value = acc40;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 4 * lanes + 1 * ldc;
        svfloat64_t value = acc41;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 4 * lanes + 2 * ldc;
        svfloat64_t value = acc42;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 4 * lanes + 3 * ldc;
        svfloat64_t value = acc43;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 5 * lanes + 0 * ldc;
        svfloat64_t value = acc50;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 5 * lanes + 1 * ldc;
        svfloat64_t value = acc51;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 5 * lanes + 2 * ldc;
        svfloat64_t value = acc52;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
    {
        double *out = C + 5 * lanes + 3 * ldc;
        svfloat64_t value = acc53;
        if (alpha != 1.0)
            value = svmul_n_f64_x(pg, value, alpha);
        svst1_f64(pg, out, svadd_f64_x(pg, svld1_f64(pg, out), value));
    }
}
#endif
