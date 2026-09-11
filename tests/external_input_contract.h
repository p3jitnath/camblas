#ifndef CAMBLAS_EXTERNAL_INPUT_CONTRACT_H
#define CAMBLAS_EXTERNAL_INPUT_CONTRACT_H

#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

/*
 * External command and sysfs text is evidence only after the complete stream
 * has been consumed. A short read is not EOF proof: stdio may have stopped
 * because of an error, or because the bounded destination is full. These
 * helpers keep those distinctions explicit and leave no partial value for a
 * caller to mistake for a valid record.
 */
static inline int camblas_external_buffer_nul_terminated(const char *buffer, size_t buffer_size)
{
    return buffer && buffer_size != 0 && memchr(buffer, '\0', buffer_size) != NULL;
}

/* Read one bounded, non-empty command result and reject any extra byte. */
static inline int camblas_external_read_stream(FILE *stream, char *buffer, size_t buffer_size,
                                               size_t *length)
{
    size_t used = 0;

    if (!stream || !buffer || buffer_size < 2 || !length)
        return -1;
    buffer[0] = '\0';
    *length = 0;
    while (used + 1 < buffer_size) {
        size_t got = fread(buffer + used, 1, buffer_size - used - 1, stream);
        if (ferror(stream))
            goto failed;
        if (got == 0) {
            if (!feof(stream))
                goto failed;
            break;
        }
        if (memchr(buffer + used, '\0', got) != NULL)
            goto failed;
        used += got;
    }
    if (used + 1 == buffer_size) {
        int extra = fgetc(stream);
        if (extra != EOF || ferror(stream) || !feof(stream))
            goto failed;
    }
    if (used == 0)
        goto failed;
    buffer[used] = '\0';
    *length = used;
    return 0;

failed:
    buffer[0] = '\0';
    *length = 0;
    return -1;
}

/* Read exactly one bounded text line, allowing EOF without a final newline. */
static inline int camblas_external_read_single_line(FILE *stream, char *buffer, size_t buffer_size)
{
    size_t length;
    int extra;

    if (!stream || !buffer || buffer_size < 2 || buffer_size > INT_MAX)
        return -1;
    memset(buffer, 0xa5, buffer_size);
    if (!fgets(buffer, (int)buffer_size, stream) || ferror(stream))
        goto failed;
    length = strnlen(buffer, buffer_size);
    if (length == buffer_size || (length + 1 < buffer_size && buffer[length + 1] != (char)0xa5))
        goto failed;
    extra = fgetc(stream);
    if (extra != EOF || ferror(stream) || !feof(stream))
        goto failed;
    length = strlen(buffer);
    while (length > 0 && (buffer[length - 1] == '\n' || buffer[length - 1] == '\r'))
        buffer[--length] = '\0';
    if (length == 0)
        goto failed;
    return 0;

failed:
    buffer[0] = '\0';
    return -1;
}

#endif /* CAMBLAS_EXTERNAL_INPUT_CONTRACT_H */
