/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Kernel self-test against plain C loops.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "selftest.h"
#include "util.h"

#define D      192
#define HEADS  3
#define HD     (D / HEADS)
#define HIDDEN 768
#define TOL    1e-4

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;

/* Uniform in [-scale, scale), xorshift64*. */
static float frand(float scale)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return scale * (float)((double)((rng_state * 0x2545f4914f6cdd1dULL) >> 11) /
                               (double)(1ULL << 53) * 2.0 -
                           1.0);
}

static float *randv(size_t n, float scale)
{
    float *v = xmalloc(sizeof(float) * n);

    for (size_t i = 0; i < n; i++)
        v[i] = frand(scale);
    return v;
}

static double gelu(double x)
{
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)));
}

/* Max |got - ref| relative to max |ref|; logs and counts a failure. */
static int check(const char *what, const float *got, const double *ref,
                 size_t n)
{
    double err = 0.0, scale = 0.0;

    for (size_t i = 0; i < n; i++) {
        double e = fabs(got[i] - ref[i]);

        if (e > err || isnan(got[i]))
            err = isnan(got[i]) ? INFINITY : e;
        if (fabs(ref[i]) > scale)
            scale = fabs(ref[i]);
    }
    err /= scale > 0.0 ? scale : 1.0;
    msg_info("  %-34s rel. error %.2e  %s", what, err,
             err <= TOL ? "ok" : "FAILED");
    return err <= TOL ? 0 : 1;
}

/* Reference Y = act(X W^T + b) (+ R). */
static double *ref_linear(const float *X, const float *W, const float *b,
                          const float *R, int rows, int in, int out, int act)
{
    double *Y = xmalloc(sizeof(double) * rows * out);

#pragma omp parallel for
    for (int r = 0; r < rows; r++)
        for (int j = 0; j < out; j++) {
            double s = b ? b[j] : 0.0;

            for (int k = 0; k < in; k++)
                s += (double)X[(size_t)r * in + k] * W[(size_t)j * in + k];
            if (act == ACT_GELU)
                s = gelu(s);
            if (R)
                s += R[(size_t)r * out + j];
            Y[(size_t)r * out + j] = s;
        }
    return Y;
}

static int test_linear(struct olmo_ops *o, const char *what, int rows, int in,
                       int out, int act, int residual)
{
    struct ocl_backend *ocl = o->ocl;
    float *X = randv((size_t)rows * in, 1.0f);
    float *W = randv((size_t)out * in, 1.0f / sqrtf(in));
    float *b = randv(out, 0.1f);
    float *R = residual ? randv((size_t)rows * out, 1.0f) : NULL;
    float *Y = xmalloc(sizeof(float) * rows * out);
    double *ref = ref_linear(X, W, b, R, rows, in, out, act);
    cl_mem dX = ocl_upload(ocl, X, sizeof(float) * rows * in, "X");
    cl_mem dW = ocl_upload(ocl, W, sizeof(float) * out * in, "W");
    cl_mem db = ocl_upload(ocl, b, sizeof(float) * out, "b");
    cl_mem dY = residual ? ocl_upload(ocl, R, sizeof(float) * rows * out, "Y")
                         : ocl_alloc(ocl, sizeof(float) * rows * out, "Y");
    int fail;

    linear(o, rows, in, out, dX, 0, dW, db, dY, 0, act, residual);
    ocl_read(ocl, dY, 0, Y, sizeof(float) * rows * out);
    fail = check(what, Y, ref, (size_t)rows * out);

    ocl_release(&dX);
    ocl_release(&dW);
    ocl_release(&db);
    ocl_release(&dY);
    free(X);
    free(W);
    free(b);
    free(R);
    free(Y);
    free(ref);
    return fail;
}

/* Multi-head attention in the encoder's layout: qkv rows hold q, k and
 * v of all heads ([T, 3D], head h at columns h * HD of each part), the
 * output rows hold the heads side by side ([T, D]). */
static int test_attention(struct olmo_ops *o, int T)
{
    struct ocl_backend *ocl = o->ocl;
    const double scale = 1.0 / sqrt((double)HD);
    float *qkv = randv((size_t)T * 3 * D, 1.0f);
    float *out = xmalloc(sizeof(float) * T * D);
    double *ref = xmalloc(sizeof(double) * T * D);
    cl_mem dqkv = ocl_upload(ocl, qkv, sizeof(float) * T * 3 * D, "qkv");
    cl_mem dS = ocl_alloc(ocl, sizeof(float) * HEADS * T * T, "scores");
    cl_mem dO = ocl_alloc(ocl, sizeof(float) * T * D, "attn out");
    int fail;

    /* S_h = Q_h K_h^T / sqrt(HD), one batch per head. */
    gemm(o, T, T, HD, bmat(dqkv, 0, 3 * D, 0, HD), bmat(dqkv, D, 3 * D, 0, HD),
         1, bmat(dS, 0, T, 0, (long)T * T), HEADS, HEADS, NULL, NULL, 0,
         (float)scale, ACT_NONE);
    softmax_rows(o, dS, (long)HEADS * T, T);
    /* O_h = S_h V_h. */
    gemm(o, T, HD, T, bmat(dS, 0, T, 0, (long)T * T),
         bmat(dqkv, 2 * D, 3 * D, 0, HD), 0, bmat(dO, 0, D, 0, HD), HEADS,
         HEADS, NULL, NULL, 0, 1.0f, ACT_NONE);
    ocl_read(ocl, dO, 0, out, sizeof(float) * T * D);

#pragma omp parallel
    {
        double *s = xmalloc(sizeof(double) * T);

#pragma omp for collapse(2)
        for (int h = 0; h < HEADS; h++)
            for (int i = 0; i < T; i++) {
                const float *q = qkv + (size_t)i * 3 * D + h * HD;
                double m = -INFINITY, sum = 0.0;

                for (int j = 0; j < T; j++) {
                    const float *k = qkv + (size_t)j * 3 * D + D + h * HD;
                    double d = 0.0;

                    for (int c = 0; c < HD; c++)
                        d += (double)q[c] * k[c];
                    s[j] = d * scale;
                    if (s[j] > m)
                        m = s[j];
                }
                for (int j = 0; j < T; j++)
                    sum += s[j] = exp(s[j] - m);
                for (int c = 0; c < HD; c++) {
                    double acc = 0.0;

                    for (int j = 0; j < T; j++)
                        acc +=
                            s[j] * qkv[(size_t)j * 3 * D + 2 * D + h * HD + c];
                    ref[(size_t)i * D + h * HD + c] = acc / sum;
                }
            }
        free(s);
    }
    fail = check("attention (3 heads, strided qkv)", out, ref, (size_t)T * D);

    ocl_release(&dqkv);
    ocl_release(&dS);
    ocl_release(&dO);
    free(qkv);
    free(out);
    free(ref);
    return fail;
}

static int test_layernorm(struct olmo_ops *o, int rows)
{
    struct ocl_backend *ocl = o->ocl;
    const float eps = 1e-5f;
    float *X = randv((size_t)rows * D, 3.0f);
    float *g = randv(D, 1.0f), *b = randv(D, 1.0f);
    float *Y = xmalloc(sizeof(float) * rows * D);
    double *ref = xmalloc(sizeof(double) * rows * D);
    cl_mem dX = ocl_upload(ocl, X, sizeof(float) * rows * D, "X");
    cl_mem dg = ocl_upload(ocl, g, sizeof(float) * D, "g");
    cl_mem db = ocl_upload(ocl, b, sizeof(float) * D, "b");
    int fail;

    /* Shift some rows far from zero: the kernel's two-pass variance must
     * not lose precision there. */
    for (int r = 0; r < rows; r += 7)
        for (int c = 0; c < D; c++)
            X[(size_t)r * D + c] += 100.0f;
    ocl_write(ocl, dX, 0, X, sizeof(float) * rows * D);
    layernorm(o, rows, D, dX, dX, dg, db, eps, ACT_NONE);
    ocl_read(ocl, dX, 0, Y, sizeof(float) * rows * D);

    for (int r = 0; r < rows; r++) {
        const float *x = X + (size_t)r * D;
        double mean = 0.0, var = 0.0;

        for (int c = 0; c < D; c++)
            mean += x[c];
        mean /= D;
        for (int c = 0; c < D; c++)
            var += (x[c] - mean) * (x[c] - mean);
        var /= D;
        for (int c = 0; c < D; c++)
            ref[(size_t)r * D + c] =
                (x[c] - mean) / sqrt(var + eps) * g[c] + b[c];
    }
    fail = check("layernorm (D=192, eps 1e-5)", Y, ref, (size_t)rows * D);

    ocl_release(&dX);
    ocl_release(&dg);
    ocl_release(&db);
    free(X);
    free(g);
    free(b);
    free(Y);
    free(ref);
    return fail;
}

int olmo_selftest(struct olmo_ops *o, int tokens)
{
    int fail = 0;

    msg_info("Self-test, %d tokens:", tokens);
    fail += test_linear(o, "linear generic (100x50->70, GELU)", 100, 50, 70,
                        ACT_GELU, 0);
    fail += test_linear(o, "linear qkv (T x 192 -> 576)", tokens, D, 3 * D,
                        ACT_NONE, 0);
    fail += test_linear(o, "linear fc1 (T x 192 -> 768, GELU)", tokens, D,
                        HIDDEN, ACT_GELU, 0);
    fail += test_linear(o, "linear fc2 (T x 768 -> 192, +res)", tokens, HIDDEN,
                        D, ACT_NONE, 1);
    fail += test_attention(o, tokens);
    fail += test_layernorm(o, tokens);
    if (fail)
        msg_warning("%d self-test check(s) failed", fail);
    else
        msg_info("All self-test checks passed");
    return fail;
}
