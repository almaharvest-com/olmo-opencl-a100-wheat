/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Messages, fatal errors and checked allocation.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

int msg_level = 1;

static void vmsg(const char *prefix, const char *fmt, va_list ap)
{
    fputs(prefix, stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
}

void fatal(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vmsg("ERROR: ", fmt, ap);
    va_end(ap);
    exit(EXIT_FAILURE);
}

void msg_warning(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vmsg("WARNING: ", fmt, ap);
    va_end(ap);
}

void msg_info(const char *fmt, ...)
{
    va_list ap;

    if (msg_level < 1)
        return;
    va_start(ap, fmt);
    vmsg("", fmt, ap);
    va_end(ap);
}

void msg_verbose(const char *fmt, ...)
{
    va_list ap;

    if (msg_level < 2)
        return;
    va_start(ap, fmt);
    vmsg("", fmt, ap);
    va_end(ap);
}

void msg_debug(const char *fmt, ...)
{
    va_list ap;

    if (msg_level < 3)
        return;
    va_start(ap, fmt);
    vmsg("D: ", fmt, ap);
    va_end(ap);
}

void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);

    if (!p)
        fatal("Out of memory allocating %zu bytes", n);
    return p;
}

void *xcalloc(size_t n, size_t size)
{
    void *p = calloc(n ? n : 1, size ? size : 1);

    if (!p)
        fatal("Out of memory allocating %zu x %zu bytes", n, size);
    return p;
}

void *xrealloc(void *p, size_t n)
{
    p = realloc(p, n ? n : 1);
    if (!p)
        fatal("Out of memory reallocating %zu bytes", n);
    return p;
}

char *xstrdup(const char *s)
{
    size_t n = strlen(s) + 1;

    return memcpy(xmalloc(n), s, n);
}

void xsnprintf(char *buf, size_t n, const char *fmt, ...)
{
    va_list ap;
    int len;

    va_start(ap, fmt);
    len = vsnprintf(buf, n, fmt, ap);
    va_end(ap);
    if (len < 0 || (size_t)len >= n)
        fatal("Path or name too long: <%.*s...>", 60, buf);
}

char *read_file(const char *path)
{
    FILE *fp = fopen(path, "rb");
    char *buf;
    long n;

    if (!fp)
        fatal("Cannot open <%s>", path);
    fseek(fp, 0, SEEK_END);
    n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    buf = xmalloc(n + 1);
    if (fread(buf, 1, n, fp) != (size_t)n)
        fatal("Cannot read <%s>", path);
    buf[n] = '\0';
    fclose(fp);
    return buf;
}

float *read_f32(const char *path, size_t n)
{
    FILE *fp = fopen(path, "rb");
    float *v;

    if (!fp)
        fatal("Cannot open <%s>", path);
    fseek(fp, 0, SEEK_END);
    if ((size_t)ftell(fp) != n * sizeof(float))
        fatal("<%s> has %ld bytes, expected %zu floats", path, ftell(fp), n);
    fseek(fp, 0, SEEK_SET);
    v = xmalloc(sizeof(float) * n);
    if (fread(v, sizeof(float), n, fp) != n)
        fatal("Cannot read <%s>", path);
    fclose(fp);
    return v;
}

/* Start of the value of member key, or NULL. */
static const char *json_value(const char *json, const char *key)
{
    size_t n = strlen(key);
    const char *p = json;

    while ((p = strchr(p, '"'))) {
        if (strncmp(p + 1, key, n) == 0 && p[n + 1] == '"') {
            p += n + 2;
            while (isspace((unsigned char)*p))
                p++;
            if (*p == ':') {
                p++;
                while (isspace((unsigned char)*p))
                    p++;
                return p;
            }
        }
        p++;
    }
    return NULL;
}

char *json_string(const char *json, const char *key)
{
    const char *p = json_value(json, key), *end;

    if (!p || *p != '"')
        return NULL;
    end = strchr(++p, '"');
    return end ? strndup(p, end - p) : NULL;
}

long json_long(const char *json, const char *key, long def)
{
    const char *p = json_value(json, key);
    char *end;
    long v;

    if (!p)
        return def;
    v = strtol(p, &end, 10);
    return end == p ? def : v;
}

int json_bool(const char *json, const char *key, int def)
{
    const char *p = json_value(json, key);

    if (!p)
        return def;
    if (strncmp(p, "true", 4) == 0)
        return 1;
    if (strncmp(p, "false", 5) == 0)
        return 0;
    return def;
}
