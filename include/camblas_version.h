/*
 * CAMBLAS public API and ABI identity.
 *
 * The current identity is deliberately a development identity: ABI major 0
 * means that binary compatibility is not promised yet.  The version macros
 * and camblas_abi_version() give applications and test harnesses one
 * fail-closed way to detect a library/header generation mismatch while the
 * public ABI is being designed.
 */
#ifndef CAMBLAS_VERSION_H
#define CAMBLAS_VERSION_H

#include <stdint.h>

/* Semantic API version for the current source interface. */
#define CAMBLAS_API_VERSION_MAJOR 0u
#define CAMBLAS_API_VERSION_MINOR 1u
#define CAMBLAS_API_VERSION_PATCH 0u
#define CAMBLAS_API_VERSION_STRING "0.1.0-dev"

/*
 * Binary ABI generation. Major zero is intentionally unstable.
 * The minor component identifies compatible development
 * snapshots only after the ABI has been frozen at major 1.
 */
#define CAMBLAS_ABI_VERSION_MAJOR 0u
#define CAMBLAS_ABI_VERSION_MINOR 2u

#define CAMBLAS_ABI_VERSION_ENCODE(major, minor) \
    ((uint32_t)((((uint32_t)(major)) << 16) | ((uint32_t)(minor))))

#define CAMBLAS_ABI_VERSION \
    CAMBLAS_ABI_VERSION_ENCODE(CAMBLAS_ABI_VERSION_MAJOR, CAMBLAS_ABI_VERSION_MINOR)

#ifdef __cplusplus
extern "C" {
#endif

/* Returns the ABI identity encoded by CAMBLAS_ABI_VERSION. */
uint32_t camblas_abi_version(void);

#ifdef __cplusplus
}
#endif

#endif /* CAMBLAS_VERSION_H */
