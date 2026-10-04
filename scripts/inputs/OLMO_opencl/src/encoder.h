/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Frozen OlmoEarth encoder forward pass on Sentinel-2 L2A
 *               crops: resize and patch extraction on the host, patch
 *               embedding and transformer blocks in OpenCL.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#ifndef OLMO_CL_ENCODER_H
#define OLMO_CL_ENCODER_H

#include <stddef.h>

#include "ops.h"
#include "weights.h"

/* Stochastic depth rate of the OlmoEarth-v1 encoders (config.json). */
#define OLMO_DROP_PATH 0.1f

/* Optional per-stage callback, for comparisons with PyTorch. Called with
 * host copies of: "resized_s<s>" [T][|s|][2c][2c], "patch" and "encoded"
 * [g][g][T][3][D], "block<i>" and "norm" [N][D]. Reading back every stage
 * is slow; pass NULL in training. */
typedef void (*olmo_trace_fn)(void *ctx, const char *stage, const float *data,
                              size_t n);

struct olmo_encoder {
    struct olmo_ops *o;
    const struct olmo_weights *w;
    int T;     /* Timesteps. */
    int crop;  /* Input crop side (64). */
    int patch; /* Requested token patch size (4). */
    int res;   /* Resized side: crop / patch * base patch (128). */
    int grid;  /* Tokens per side: crop / patch (16). */
    int ntok;  /* grid * grid * T * bandsets. */
    float *rw; /* [res][crop] bicubic resize weights. */
    float *resized[OLMO_NBANDSETS]; /* Host [T][|s|][res][res]. */
    float *cols[OLMO_NBANDSETS];    /* Host im2col [T][grid^2][|s| 64]. */
    float *tokens;                  /* Host [ntok][D] readback. */
    cl_mem d_cols[OLMO_NBANDSETS];
    cl_mem d_patch_w[OLMO_NBANDSETS], d_patch_b[OLMO_NBANDSETS];
    cl_mem d_enc; /* [ntok][D] encoding table, constant. */
    cl_mem d_ln1_w[OLMO_MAX_DEPTH], d_ln1_b[OLMO_MAX_DEPTH];
    cl_mem d_qkv_w[OLMO_MAX_DEPTH], d_qkv_b[OLMO_MAX_DEPTH];
    cl_mem d_proj_w[OLMO_MAX_DEPTH], d_proj_b[OLMO_MAX_DEPTH];
    cl_mem d_ln2_w[OLMO_MAX_DEPTH], d_ln2_b[OLMO_MAX_DEPTH];
    cl_mem d_fc1_w[OLMO_MAX_DEPTH], d_fc1_b[OLMO_MAX_DEPTH];
    cl_mem d_fc2_w[OLMO_MAX_DEPTH], d_fc2_b[OLMO_MAX_DEPTH];
    /* Biases divided by the keep probability, for kept branches under
     * stochastic depth. */
    cl_mem d_proj_b_dp[OLMO_MAX_DEPTH], d_fc2_b_dp[OLMO_MAX_DEPTH];
    cl_mem d_norm_w, d_norm_b;
    cl_mem X, XN, QKV, S, ATT, H; /* Activations. */
};

/* Upload the weights and allocate the buffers for T timesteps of
 * crop x crop pixels with token patch size patch. */
void encoder_init(struct olmo_encoder *e, struct olmo_ops *o,
                  const struct olmo_weights *w, int T, int crop, int patch);

void encoder_free(struct olmo_encoder *e);

/* Encode one normalised crop x [T][12][crop][crop] into the pooled
 * feature map features [D][grid][grid] (mean over timesteps and band
 * sets, as rslearn's OlmoEarth wrapper).
 *
 * keep, when not NULL, applies stochastic depth as in training mode:
 * keep[2 i] and keep[2 i + 1] tell whether the attention and MLP
 * branches of block i are kept (then scaled by 1 / (1 - p)) or skipped.
 * NULL is evaluation mode. */
void encoder_forward(struct olmo_encoder *e, const float *x,
                     const unsigned char *keep, float *features,
                     olmo_trace_fn trace, void *trace_ctx);

#endif /* OLMO_CL_ENCODER_H */
