#define _GNU_SOURCE 1

#include "external_input_contract.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

static int expect(const char *name, int actual, int wanted)
{
    if (actual == wanted)
        return 0;
    fprintf(stderr, "FAIL %s: got %d, expected %d\n", name, actual, wanted);
    return 1;
}

static int expect_text(const char *name, const char *actual, const char *wanted)
{
    if (actual && wanted && strcmp(actual, wanted) == 0)
        return 0;
    fprintf(stderr, "FAIL %s: got <%s>, expected <%s>\n", name, actual ? actual : "(null)",
            wanted ? wanted : "(null)");
    return 1;
}

static FILE *stream_from_text(const char *text)
{
    FILE *stream = tmpfile();
    size_t length;
    if (!stream)
        return NULL;
    length = strlen(text);
    if (fwrite(text, 1, length, stream) != length || fflush(stream) != 0 ||
        fseek(stream, 0, SEEK_SET) != 0) {
        fclose(stream);
        return NULL;
    }
    return stream;
}

#ifdef __GLIBC__
static ssize_t failing_read(void *cookie, char *buffer, size_t size)
{
    (void)cookie;
    (void)buffer;
    (void)size;
    errno = EIO;
    return -1;
}

static int no_op_close(void *cookie)
{
    (void)cookie;
    return 0;
}

static FILE *error_stream(void)
{
    cookie_io_functions_t functions = {
        .read = failing_read,
        .write = NULL,
        .seek = NULL,
        .close = no_op_close,
    };
    return fopencookie(NULL, "r", functions);
}
#endif

int main(void)
{
    char buffer[128];
    size_t length = 0;
    int failures = 0;

    {
        FILE *stream = stream_from_text("JobId=5959000 JobState=RUNNING");
        if (!stream) {
            fprintf(stderr, "FAIL command stream setup: %s\n", strerror(errno));
            failures++;
        } else {
            failures +=
                expect("complete command stream",
                       camblas_external_read_stream(stream, buffer, sizeof(buffer), &length), 0);
            failures +=
                expect_text("complete command value", buffer, "JobId=5959000 JobState=RUNNING");
            failures += expect("complete command length", length == strlen(buffer), 1);
            fclose(stream);
        }
    }

    {
        FILE *stream = stream_from_text("0123456789");
        char small[8];
        if (!stream) {
            fprintf(stderr, "FAIL truncation stream setup: %s\n", strerror(errno));
            failures++;
        } else {
            failures +=
                expect("oversized command stream rejected",
                       camblas_external_read_stream(stream, small, sizeof(small), &length), -1);
            failures +=
                expect("oversized command leaves no value", small[0] == '\0' && length == 0, 1);
            fclose(stream);
        }
    }

    {
        FILE *stream = stream_from_text("");
        if (!stream) {
            fprintf(stderr, "FAIL empty stream setup: %s\n", strerror(errno));
            failures++;
        } else {
            failures +=
                expect("empty command stream rejected",
                       camblas_external_read_stream(stream, buffer, sizeof(buffer), &length), -1);
            fclose(stream);
        }
    }

#ifdef __GLIBC__
    {
        FILE *stream = error_stream();
        if (!stream) {
            fprintf(stderr, "FAIL error stream setup: %s\n", strerror(errno));
            failures++;
        } else {
            failures +=
                expect("fread error propagated",
                       camblas_external_read_stream(stream, buffer, sizeof(buffer), &length), -1);
            failures += expect("fread error leaves no value", buffer[0] == '\0' && length == 0, 1);
            fclose(stream);
        }
    }
#endif

    {
        FILE *stream = stream_from_text("performance\n");
        if (!stream) {
            fprintf(stderr, "FAIL governor stream setup: %s\n", strerror(errno));
            failures++;
        } else {
            failures +=
                expect("newline-terminated sysfs line",
                       camblas_external_read_single_line(stream, buffer, sizeof(buffer)), 0);
            failures += expect_text("newline-terminated governor", buffer, "performance");
            fclose(stream);
        }
    }

    {
        FILE *stream = stream_from_text("powersave");
        if (!stream) {
            fprintf(stderr, "FAIL no-newline stream setup: %s\n", strerror(errno));
            failures++;
        } else {
            failures +=
                expect("EOF-terminated sysfs line",
                       camblas_external_read_single_line(stream, buffer, sizeof(buffer)), 0);
            failures += expect_text("EOF-terminated governor", buffer, "powersave");
            fclose(stream);
        }
    }

    {
        FILE *stream = stream_from_text("performance\nextra\n");
        if (!stream) {
            fprintf(stderr, "FAIL trailing-data stream setup: %s\n", strerror(errno));
            failures++;
        } else {
            failures +=
                expect("trailing sysfs data rejected",
                       camblas_external_read_single_line(stream, buffer, sizeof(buffer)), -1);
            fclose(stream);
        }
    }

    {
        FILE *stream = stream_from_text("performance\n");
        char small[8];
        if (!stream) {
            fprintf(stderr, "FAIL line truncation stream setup: %s\n", strerror(errno));
            failures++;
        } else {
            failures += expect("truncated sysfs line rejected",
                               camblas_external_read_single_line(stream, small, sizeof(small)), -1);
            failures += expect("truncated sysfs line leaves no value", small[0] == '\0', 1);
            fclose(stream);
        }
    }

#ifdef __GLIBC__
    {
        FILE *stream = error_stream();
        if (!stream) {
            fprintf(stderr, "FAIL line error stream setup: %s\n", strerror(errno));
            failures++;
        } else {
            failures +=
                expect("fgets error propagated",
                       camblas_external_read_single_line(stream, buffer, sizeof(buffer)), -1);
            fclose(stream);
        }
    }
#endif

    {
        char terminated[] = {'n', 'i', 'd', '\0'};
        char unterminated[] = {'n', 'i', 'd', 'x'};
        failures +=
            expect("bounded hostname terminator",
                   camblas_external_buffer_nul_terminated(terminated, sizeof(terminated)), 1);
        failures +=
            expect("hostname truncation detected",
                   camblas_external_buffer_nul_terminated(unterminated, sizeof(unterminated)), 0);
        failures += expect("null hostname buffer rejected",
                           camblas_external_buffer_nul_terminated(NULL, 4), 0);
    }

    if (failures != 0)
        return 1;
    puts("external input contract PASS: stream errors/truncation/hostname boundaries");
    return 0;
}
