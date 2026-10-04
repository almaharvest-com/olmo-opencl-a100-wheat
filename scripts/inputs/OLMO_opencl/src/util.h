/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Messages, fatal errors and checked allocation, standing in
 *               for the GRASS GIS library calls of the code reused from
 *               i.sam.opencl.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#ifndef OLMO_CL_UTIL_H
#define OLMO_CL_UTIL_H

#include <stddef.h>

/* No translation catalogue: the reused sources keep their _() wrappers so
 * that they stay easy to diff against i.sam.opencl. */
#define _(s) (s)

/* 0 = quiet, 1 = normal (default), 2 = verbose, 3 = debug. Set from the
 * command line or the OLMO_CL_VERBOSE environment variable. */
extern int msg_level;

void fatal(const char *fmt, ...)
    __attribute__((format(printf, 1, 2), noreturn));
void msg_warning(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void msg_info(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void msg_verbose(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void msg_debug(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

void *xmalloc(size_t n);
void *xcalloc(size_t n, size_t size);
void *xrealloc(void *p, size_t n);
char *xstrdup(const char *s);

/* Minimal JSON lookups for flat metadata files: the first member called
 * key anywhere in the text. json_string() returns a new string or NULL;
 * json_long() and json_bool() return def when the key is absent. */
char *json_string(const char *json, const char *key);
long json_long(const char *json, const char *key, long def);
int json_bool(const char *json, const char *key, int def);

/* Whole file as a NUL-terminated string, fatal on error. */
char *read_file(const char *path);

/* Whole file of n float32 values, fatal if its size differs. */
float *read_f32(const char *path, size_t n);

/* snprintf() into buf of size n, fatal if the result does not fit. */
void xsnprintf(char *buf, size_t n, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

#endif /* OLMO_CL_UTIL_H */
