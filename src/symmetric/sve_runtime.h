/*
 * CAMBLAS internal SVE runtime metadata helpers.
 *
 * Linux PR_SVE_GET_VL returns the current vector length in bytes together
 * with the PR_SVE_VL_INHERIT flag.  Keep decoding in one private helper so
 * topology discovery, the SVE library, and the benchmark harness cannot
 * disagree about what constitutes a usable value.
 */
#ifndef CAMBLAS_INTERNAL_SVE_RUNTIME_H
#define CAMBLAS_INTERNAL_SVE_RUNTIME_H

#if defined(__aarch64__)

#ifndef PR_SVE_GET_VL
#define PR_SVE_GET_VL 51
#endif

#ifndef PR_SVE_VL_LEN_MASK
#define PR_SVE_VL_LEN_MASK 0xffff
#endif

#ifndef PR_SVE_VL_INHERIT
#define PR_SVE_VL_INHERIT (1 << 17)
#endif

/* Linux UAPI SVE_VL_MIN/MAX are one through 512 128-bit quadwords. */
#define CAMBLAS_SVE_VL_MIN_BYTES 16
#define CAMBLAS_SVE_VL_MAX_BYTES 8192
#define CAMBLAS_SVE_VL_GRANULE_BYTES 16

/*
 * Decode one PR_SVE_GET_VL return value.  Zero means unavailable or
 * malformed.  GET may report only PR_SVE_VL_INHERIT; SET_VL_ONEXEC and any
 * other currently unknown high bit are rejected rather than silently
 * discarded.  The length must satisfy the Linux UAPI granularity and bounds.
 */
static inline int camblas_sve_vl_bits_from_prctl(int result)
{
    unsigned int raw;
    unsigned int length;
    unsigned int allowed;

    if (result < 0)
        return 0;
    raw = (unsigned int)result;
    allowed = (unsigned int)PR_SVE_VL_LEN_MASK | (unsigned int)PR_SVE_VL_INHERIT;
    if ((raw & ~allowed) != 0U)
        return 0;

    length = raw & (unsigned int)PR_SVE_VL_LEN_MASK;
    if (length < CAMBLAS_SVE_VL_MIN_BYTES || length > CAMBLAS_SVE_VL_MAX_BYTES ||
        length % CAMBLAS_SVE_VL_GRANULE_BYTES != 0U)
        return 0;

    return (int)(length * 8U);
}

#else

static inline int camblas_sve_vl_bits_from_prctl(int result)
{
    (void)result;
    return 0;
}

#endif

#endif /* CAMBLAS_INTERNAL_SVE_RUNTIME_H */
