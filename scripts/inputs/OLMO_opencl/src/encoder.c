/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Frozen OlmoEarth encoder forward pass on Sentinel-2 L2A
 *               crops.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "encoder.h"
#include "util.h"

/* Token order of the flattened sequence, as OlmoEarth's
 * collapse_and_combine_hwtc: row ((h * grid + w) * T + t) * S + s. */
#define TOKEN(e, h, w, t, s) \
    ((((h) * (e)->grid + (w)) * (e)->T + (t)) * OLMO_NBANDSETS + (s))

/* PyTorch's antialiased bicubic filter (_upsample_bicubic2d_aa), which
 * uses the PIL coefficient a = -0.5, not the -0.75 of plain bicubic. */
static double aa_cubic(double x)
{
    const double a = -0.5;

    x = fabs(x);
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0)
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

/* Dense [out][in] weights of F.interpolate(mode="bicubic",
 * antialias=True, align_corners=False) along one axis: taps within the
 * support, truncated at the borders and renormalised. */
static float *resize_weights(int in, int out)
{
    const double scale = (double)in / out;
    const double support = scale >= 1.0 ? 2.0 * scale : 2.0;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    float *wt = xcalloc((size_t)in * out, sizeof(float));

    for (int i = 0; i < out; i++) {
        const double center = scale * (i + 0.5);
        /* Truncation toward zero, as the C++ static_cast. */
        long lo = (long)(center - support + 0.5), hi;
        double sum = 0.0, v[64];

        lo = lo < 0 ? 0 : lo;
        hi = (long)(center + support + 0.5);
        hi = hi > in ? in : hi;
        if (hi - lo > 64)
            fatal("Resize %d -> %d needs more than 64 taps", in, out);
        for (long j = lo; j < hi; j++)
            sum += v[j - lo] = aa_cubic((j - center + 0.5) * invscale);
        for (long j = lo; j < hi; j++)
            wt[(size_t)i * in + j] = (float)(v[j - lo] / sum);
    }
    return wt;
}

/* Resize one crop x crop plane to res x res with the dense weights. */
static void resize_plane(const float *wt, int crop, int res, const float *in,
                         float *out, float *tmp)
{
    /* Along x, then along y. */
    for (int r = 0; r < crop; r++)
        for (int c = 0; c < res; c++) {
            const float *wr = wt + (size_t)c * crop;
            float acc = 0.0f;

            for (int j = 0; j < crop; j++)
                acc += wr[j] * in[r * crop + j];
            tmp[r * res + c] = acc;
        }
    for (int r = 0; r < res; r++) {
        const float *wr = wt + (size_t)r * crop;

        for (int c = 0; c < res; c++) {
            float acc = 0.0f;

            for (int j = 0; j < crop; j++)
                acc += wr[j] * tmp[j * res + c];
            out[r * res + c] = acc;
        }
    }
}

/* 1D sin-cos position encoding of OlmoEarth (nn/encodings.py), dim
 * values: frequencies 1 / 10000^(i / dim / 2), sines then cosines. */
static void sincos_1d(double pos, int dim, float *out)
{
    for (int i = 0; i < dim / 2; i++) {
        const double omega = 1.0 / pow(10000.0, (double)i / dim / 2.0);

        out[i] = (float)sin(pos * omega);
        out[dim / 2 + i] = (float)cos(pos * omega);
    }
}

/* Encoding added to every token, independent of the input: four slots
 * of D / 4: band-set embedding, time position, month (legacy timestamps:
 * month = t) and 2D spatial position. The spatial slot encodes the
 * column first, then the row, both scaled by the GSD ratio
 * input_res * patch / BASE_GSD = patch. */
static float *encoding_table(const struct olmo_encoder *e)
{
    const struct olmo_weights *w = e->w;
    const int D = w->dim, n = w->enc_dim;
    float *E = xcalloc((size_t)e->ntok * D, sizeof(float));

    if (e->T > w->max_seqlen || e->T > 12)
        fatal("%d timesteps exceed the encoder's %d time positions", e->T,
              w->max_seqlen < 12 ? w->max_seqlen : 12);
    for (int h = 0; h < e->grid; h++)
        for (int x = 0; x < e->grid; x++)
            for (int t = 0; t < e->T; t++)
                for (int s = 0; s < OLMO_NBANDSETS; s++) {
                    float *r = E + (size_t)TOKEN(e, h, x, t, s) * D;

                    memcpy(r, w->channel_embed + (size_t)s * n,
                           sizeof(float) * n);
                    memcpy(r + n, w->pos_embed + (size_t)t * n,
                           sizeof(float) * n);
                    memcpy(r + 2 * n, w->month_embed + (size_t)t * n,
                           sizeof(float) * n);
                    sincos_1d((double)x * e->patch, n / 2, r + 3 * n);
                    sincos_1d((double)h * e->patch, n / 2, r + 3 * n + n / 2);
                }
    return E;
}

static cl_mem up(struct olmo_encoder *e, const float *src, size_t n,
                 const char *what)
{
    return ocl_upload(e->o->ocl, src, sizeof(float) * n, what);
}

static cl_mem up_scaled(struct olmo_encoder *e, const float *src, size_t n,
                        float f, const char *what)
{
    float *tmp = xmalloc(sizeof(float) * n);
    cl_mem m;

    for (size_t i = 0; i < n; i++)
        tmp[i] = src[i] * f;
    m = up(e, tmp, n, what);
    free(tmp);
    return m;
}

void encoder_init(struct olmo_encoder *e, struct olmo_ops *o,
                  const struct olmo_weights *w, int T, int crop, int patch)
{
    struct ocl_backend *ocl = o->ocl;
    const int D = w->dim, H = w->hidden;
    const float kinv = 1.0f / (1.0f - OLMO_DROP_PATH);
    float *E;
    size_t ntok, scores;

    memset(e, 0, sizeof(*e));
    if (crop % patch || patch > OLMO_BASE_PATCH)
        fatal("Crop %d is not a multiple of patch size %d (at most %d)", crop,
              patch, OLMO_BASE_PATCH);
    e->o = o;
    e->w = w;
    e->T = T;
    e->crop = crop;
    e->patch = patch;
    e->grid = crop / patch;
    e->res = e->grid * OLMO_BASE_PATCH;
    e->ntok = e->grid * e->grid * T * OLMO_NBANDSETS;
    ntok = e->ntok;
    e->rw = resize_weights(crop, e->res);

    for (int s = 0; s < OLMO_NBANDSETS; s++) {
        const size_t k =
            (size_t)olmo_bandset_size[s] * OLMO_BASE_PATCH * OLMO_BASE_PATCH;

        e->resized[s] =
            xmalloc(sizeof(float) * T * olmo_bandset_size[s] * e->res * e->res);
        e->cols[s] = xmalloc(sizeof(float) * T * e->grid * e->grid * k);
        e->d_cols[s] =
            ocl_alloc(ocl, sizeof(float) * T * e->grid * e->grid * k, "im2col");
        e->d_patch_w[s] = up(e, w->patch_w[s], (size_t)D * k, "patch_w");
        e->d_patch_b[s] = up(e, w->patch_b[s], D, "patch_b");
    }
    E = encoding_table(e);
    e->d_enc = up(e, E, ntok * D, "encodings");
    free(E);

    for (int b = 0; b < w->depth; b++) {
        const struct olmo_block *k = &w->blocks[b];

        e->d_ln1_w[b] = up(e, k->ln1_w, D, "ln1_w");
        e->d_ln1_b[b] = up(e, k->ln1_b, D, "ln1_b");
        e->d_qkv_w[b] = up(e, k->qkv_w, (size_t)3 * D * D, "qkv_w");
        e->d_qkv_b[b] = up(e, k->qkv_b, (size_t)3 * D, "qkv_b");
        e->d_proj_w[b] = up(e, k->proj_w, (size_t)D * D, "proj_w");
        e->d_proj_b[b] = up(e, k->proj_b, D, "proj_b");
        e->d_proj_b_dp[b] = up_scaled(e, k->proj_b, D, kinv, "proj_b_dp");
        e->d_ln2_w[b] = up(e, k->ln2_w, D, "ln2_w");
        e->d_ln2_b[b] = up(e, k->ln2_b, D, "ln2_b");
        e->d_fc1_w[b] = up(e, k->fc1_w, (size_t)H * D, "fc1_w");
        e->d_fc1_b[b] = up(e, k->fc1_b, H, "fc1_b");
        e->d_fc2_w[b] = up(e, k->fc2_w, (size_t)D * H, "fc2_w");
        e->d_fc2_b[b] = up(e, k->fc2_b, D, "fc2_b");
        e->d_fc2_b_dp[b] = up_scaled(e, k->fc2_b, D, kinv, "fc2_b_dp");
    }
    e->d_norm_w = up(e, w->norm_w, D, "norm_w");
    e->d_norm_b = up(e, w->norm_b, D, "norm_b");

    e->X = ocl_alloc(ocl, sizeof(float) * ntok * D, "tokens");
    e->XN = ocl_alloc(ocl, sizeof(float) * ntok * D, "normed tokens");
    e->QKV = ocl_alloc(ocl, sizeof(float) * ntok * 3 * D, "qkv");
    e->ATT = ocl_alloc(ocl, sizeof(float) * ntok * D, "attention output");
    e->H = ocl_alloc(ocl, sizeof(float) * ntok * H, "mlp hidden");
    /* One score matrix per head; ocl_alloc fails loudly if the device
     * cannot hold them (255 MB for 4608 tokens and 3 heads). */
    scores = (size_t)w->heads * ntok * ntok;
    if (scores >= (size_t)1 << 31)
        fatal("%zu attention scores exceed the kernels' 32-bit indexing",
              scores);
    e->S = ocl_alloc(ocl, sizeof(float) * scores, "attention scores");
    e->tokens = xmalloc(sizeof(float) * ntok * D);

    msg_verbose("Encoder: %d timesteps, %dx%d crop, patch %d, %d tokens, "
                "%zu MiB of attention scores",
                T, crop, crop, patch, e->ntok, (sizeof(float) * scores) >> 20);
}

void encoder_free(struct olmo_encoder *e)
{
    for (int s = 0; s < OLMO_NBANDSETS; s++) {
        free(e->resized[s]);
        free(e->cols[s]);
        ocl_release(&e->d_cols[s]);
        ocl_release(&e->d_patch_w[s]);
        ocl_release(&e->d_patch_b[s]);
    }
    for (int b = 0; b < e->w->depth; b++) {
        ocl_release(&e->d_ln1_w[b]);
        ocl_release(&e->d_ln1_b[b]);
        ocl_release(&e->d_qkv_w[b]);
        ocl_release(&e->d_qkv_b[b]);
        ocl_release(&e->d_proj_w[b]);
        ocl_release(&e->d_proj_b[b]);
        ocl_release(&e->d_proj_b_dp[b]);
        ocl_release(&e->d_ln2_w[b]);
        ocl_release(&e->d_ln2_b[b]);
        ocl_release(&e->d_fc1_w[b]);
        ocl_release(&e->d_fc1_b[b]);
        ocl_release(&e->d_fc2_w[b]);
        ocl_release(&e->d_fc2_b[b]);
        ocl_release(&e->d_fc2_b_dp[b]);
    }
    ocl_release(&e->d_enc);
    ocl_release(&e->d_norm_w);
    ocl_release(&e->d_norm_b);
    ocl_release(&e->X);
    ocl_release(&e->XN);
    ocl_release(&e->QKV);
    ocl_release(&e->ATT);
    ocl_release(&e->H);
    ocl_release(&e->S);
    free(e->rw);
    free(e->tokens);
    memset(e, 0, sizeof(*e));
}

/* Host side: resize every band set and lay the patches out as GEMM rows
 * [t][patch][channel, ky, kx], matching the conv weight layout. */
static void prepare_patches(struct olmo_encoder *e, const float *x,
                            olmo_trace_fn trace, void *ctx)
{
    const int crop = e->crop, res = e->res, g = e->grid, P = OLMO_BASE_PATCH;
    const size_t in_plane = (size_t)crop * crop, plane = (size_t)res * res;

    for (int s = 0; s < OLMO_NBANDSETS; s++) {
        const int nc = olmo_bandset_size[s], c0 = olmo_bandset_first[s];
        const int k = nc * P * P;

#pragma omp parallel
        {
            float *tmp = xmalloc(sizeof(float) * crop * res);

#pragma omp for collapse(2)
            for (int t = 0; t < e->T; t++)
                for (int c = 0; c < nc; c++) {
                    const float *src =
                        x + ((size_t)t * OLMO_NBANDS + c0 + c) * in_plane;
                    float *dst = e->resized[s] + ((size_t)t * nc + c) * plane;

                    if (res == crop)
                        memcpy(dst, src, sizeof(float) * plane);
                    else
                        resize_plane(e->rw, crop, res, src, dst, tmp);
                }
            free(tmp);
        }
        if (trace) {
            char name[32];

            snprintf(name, sizeof(name), "resized_s%d", s);
            trace(ctx, name, e->resized[s], (size_t)e->T * nc * plane);
        }

#pragma omp parallel for collapse(2)
        for (int t = 0; t < e->T; t++)
            for (int p = 0; p < g * g; p++) {
                const int ph = p / g, pw = p % g;
                float *row = e->cols[s] + ((size_t)t * g * g + p) * k;

                for (int c = 0; c < nc; c++)
                    for (int ky = 0; ky < P; ky++)
                        memcpy(row + (c * P + ky) * P,
                               e->resized[s] + ((size_t)t * nc + c) * plane +
                                   (size_t)(ph * P + ky) * res + pw * P,
                               sizeof(float) * P);
            }
        ocl_write(e->o->ocl, e->d_cols[s], 0, e->cols[s],
                  sizeof(float) * e->T * g * g * k);
    }
}

static void trace_tokens(struct olmo_encoder *e, cl_mem buf, const char *name,
                         olmo_trace_fn trace, void *ctx)
{
    const size_t n = (size_t)e->ntok * e->w->dim;

    ocl_read(e->o->ocl, buf, 0, e->tokens, sizeof(float) * n);
    trace(ctx, name, e->tokens, n);
}

/* x += proj(attention(LN1(x))), heads batched through strided GEMMs. */
static void attention_branch(struct olmo_encoder *e, int b, float alpha,
                             cl_mem proj_b)
{
    struct olmo_ops *o = e->o;
    const int D = e->w->dim, N = e->ntok, nh = e->w->heads, hd = D / nh;
    const long plane = (long)N * N;

    layernorm(o, N, D, e->X, e->XN, e->d_ln1_w[b], e->d_ln1_b[b], 1e-5f,
              ACT_NONE);
    linear(o, N, D, 3 * D, e->XN, 0, e->d_qkv_w[b], e->d_qkv_b[b], e->QKV, 0,
           ACT_NONE, 0);
    /* S_h = Q_h K_h^T / sqrt(hd); q, k, v of head h sit at columns
     * h * hd of the three D-wide parts of each qkv row. */
    gemm(o, N, N, hd, bmat(e->QKV, 0, 3 * D, 0, hd),
         bmat(e->QKV, D, 3 * D, 0, hd), 1, bmat(e->S, 0, N, 0, plane), nh, nh,
         NULL, NULL, 0, 1.0f / sqrtf((float)hd), ACT_NONE);
    softmax_rows(o, e->S, (long)nh * N, N);
    /* O_h = S_h V_h, heads side by side in the output rows. */
    gemm(o, N, hd, N, bmat(e->S, 0, N, 0, plane),
         bmat(e->QKV, 2 * D, 3 * D, 0, hd), 0, bmat(e->ATT, 0, D, 0, hd), nh,
         nh, NULL, NULL, 0, 1.0f, ACT_NONE);
    gemm(o, N, D, D, mat(e->ATT, 0, D), mat(e->d_proj_w[b], 0, D), 1,
         mat(e->X, 0, D), 1, 1, proj_b, e->X, 0, alpha, ACT_NONE);
}

/* x += fc2(GELU(fc1(LN2(x)))). */
static void mlp_branch(struct olmo_encoder *e, int b, float alpha, cl_mem fc2_b)
{
    struct olmo_ops *o = e->o;
    const int D = e->w->dim, Hd = e->w->hidden, N = e->ntok;

    layernorm(o, N, D, e->X, e->XN, e->d_ln2_w[b], e->d_ln2_b[b], 1e-5f,
              ACT_NONE);
    linear(o, N, D, Hd, e->XN, 0, e->d_fc1_w[b], e->d_fc1_b[b], e->H, 0,
           ACT_GELU, 0);
    gemm(o, N, D, Hd, mat(e->H, 0, Hd), mat(e->d_fc2_w[b], 0, Hd), 1,
         mat(e->X, 0, D), 1, 1, fc2_b, e->X, 0, alpha, ACT_NONE);
}

void encoder_forward(struct olmo_encoder *e, const float *x,
                     const unsigned char *keep, float *features,
                     olmo_trace_fn trace, void *ctx)
{
    struct olmo_ops *o = e->o;
    const struct olmo_weights *w = e->w;
    const int D = w->dim, g = e->grid, P = OLMO_BASE_PATCH;
    const float kinv = 1.0f / (1.0f - OLMO_DROP_PATH);

    prepare_patches(e, x, trace, ctx);

    /* Patch embedding: for band set s and timestep t, the g * g patch
     * rows land at tokens (p, t, s), i.e. row stride T * S * D. */
    for (int s = 0; s < OLMO_NBANDSETS; s++) {
        const int k = olmo_bandset_size[s] * P * P;

        gemm(o, g * g, D, k, bmat(e->d_cols[s], 0, k, 0, (long)g * g * k),
             mat(e->d_patch_w[s], 0, k), 1,
             bmat(e->X, (long)s * D, e->T * OLMO_NBANDSETS * D, 0,
                  (long)OLMO_NBANDSETS * D),
             e->T, e->T, e->d_patch_b[s], NULL, 0, 1.0f, ACT_NONE);
    }
    if (trace)
        trace_tokens(e, e->X, "patch", trace, ctx);
    add_inplace(o, e->X, e->d_enc, e->ntok * D);
    if (trace)
        trace_tokens(e, e->X, "encoded", trace, ctx);

    for (int b = 0; b < w->depth; b++) {
        /* Stochastic depth: a dropped branch is skipped, a kept one is
         * scaled by 1 / keep probability (bias included). */
        if (!keep)
            attention_branch(e, b, 1.0f, e->d_proj_b[b]);
        else if (keep[2 * b])
            attention_branch(e, b, kinv, e->d_proj_b_dp[b]);
        if (!keep)
            mlp_branch(e, b, 1.0f, e->d_fc2_b[b]);
        else if (keep[2 * b + 1])
            mlp_branch(e, b, kinv, e->d_fc2_b_dp[b]);
        if (trace) {
            char name[32];

            snprintf(name, sizeof(name), "block%d", b);
            trace_tokens(e, e->X, name, trace, ctx);
        }
    }

    layernorm(o, e->ntok, D, e->X, e->XN, e->d_norm_w, e->d_norm_b, 1e-5f,
              ACT_NONE);
    ocl_read(o->ocl, e->XN, 0, e->tokens, sizeof(float) * e->ntok * D);
    if (trace)
        trace(ctx, "norm", e->tokens, (size_t)e->ntok * D);

    /* Mean over timesteps and band sets, to [D][g][g]. */
#pragma omp parallel for collapse(2)
    for (int h = 0; h < g; h++)
        for (int x0 = 0; x0 < g; x0++) {
            const int n = e->T * OLMO_NBANDSETS;

            for (int c = 0; c < D; c++) {
                double acc = 0.0;

                for (int t = 0; t < e->T; t++)
                    for (int s = 0; s < OLMO_NBANDSETS; s++)
                        acc += e->tokens[(size_t)TOKEN(e, h, x0, t, s) * D + c];
                features[((size_t)c * g + h) * g + x0] = (float)(acc / n);
            }
        }
}
