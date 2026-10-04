/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Frozen OlmoEarth encoder weights for Sentinel-2 L2A.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pth_loader.h"
#include "util.h"
#include "weights.h"

const int olmo_bandset_first[OLMO_NBANDSETS] = {0, 4, 10};
const int olmo_bandset_size[OLMO_NBANDSETS] = {4, 6, 2};

#define PREFIX "encoder."
#define S2     "sentinel2_l2a"

/* Read the tensor PREFIX name, checking its shape; -1 in shape matches
 * any size. */
static float *get(const struct pth_file *pf, const char *name, int ndim,
                  long s0, long s1, long s2, long s3)
{
    const long shape[4] = {s0, s1, s2, s3};
    char full[256];

    snprintf(full, sizeof(full), PREFIX "%s", name);
    return pth_read_f32(pf, pth_require(pf, full, ndim, shape));
}

/* Stack q, k and v of one block into a single [3D, D] projection so that
 * one GEMM computes all three. */
static void load_qkv(const struct pth_file *pf, int b, int D,
                     struct olmo_block *blk)
{
    static const char *part[3] = {"q", "k", "v"};
    char name[128];

    blk->qkv_w = xmalloc(sizeof(float) * 3 * D * D);
    blk->qkv_b = xmalloc(sizeof(float) * 3 * D);
    for (int i = 0; i < 3; i++) {
        float *t;

        snprintf(name, sizeof(name), "blocks.%d.attn.%s.weight", b, part[i]);
        t = get(pf, name, 2, D, D, 0, 0);
        memcpy(blk->qkv_w + (size_t)i * D * D, t, sizeof(float) * D * D);
        free(t);
        snprintf(name, sizeof(name), "blocks.%d.attn.%s.bias", b, part[i]);
        t = get(pf, name, 1, D, 0, 0, 0);
        memcpy(blk->qkv_b + (size_t)i * D, t, sizeof(float) * D);
        free(t);
    }
}

void olmo_weights_load(struct olmo_weights *w, const char *path, int heads)
{
    const struct pth_tensor *t;
    struct pth_file pf;
    char name[160];
    int D, H;

    memset(w, 0, sizeof(*w));
    pth_open(&pf, path);

    /* Sizes are taken from the tensors themselves, then every other
     * tensor is checked against them. */
    t = pth_require(&pf, PREFIX "norm.weight", 1, NULL);
    D = w->dim = (int)t->shape[0];
    t = pth_require(&pf, PREFIX "blocks.0.mlp.fc1.weight", 2, NULL);
    H = w->hidden = (int)t->shape[0];
    while (w->depth < OLMO_MAX_DEPTH) {
        snprintf(name, sizeof(name), PREFIX "blocks.%d.norm1.weight", w->depth);
        if (!pth_find(&pf, name))
            break;
        w->depth++;
    }
    if (w->depth == 0)
        fatal("No encoder blocks in <%s>", path);
    if (heads < 1 || D % heads)
        fatal("Embedding size %d is not divisible into %d heads", D, heads);
    if (D % 4)
        fatal("Embedding size %d is not divisible by 4", D);
    w->heads = heads;
    w->enc_dim = D / 4;

    for (int s = 0; s < OLMO_NBANDSETS; s++) {
        snprintf(name, sizeof(name),
                 "patch_embeddings.per_modality_embeddings." S2 "." S2
                 "__%d.proj.weight",
                 s);
        w->patch_w[s] = get(&pf, name, 4, D, olmo_bandset_size[s],
                            OLMO_BASE_PATCH, OLMO_BASE_PATCH);
        snprintf(name, sizeof(name),
                 "patch_embeddings.per_modality_embeddings." S2 "." S2
                 "__%d.proj.bias",
                 s);
        w->patch_b[s] = get(&pf, name, 1, D, 0, 0, 0);
    }

    w->channel_embed =
        get(&pf, "composite_encodings.per_modality_channel_embeddings." S2, 2,
            OLMO_NBANDSETS, w->enc_dim, 0, 0);
    t = pth_require(&pf, PREFIX "composite_encodings.pos_embed", 2, NULL);
    w->max_seqlen = (int)t->shape[0];
    w->pos_embed =
        get(&pf, "composite_encodings.pos_embed", 2, -1, w->enc_dim, 0, 0);
    w->month_embed = get(&pf, "composite_encodings.month_embed.weight", 2, 12,
                         w->enc_dim, 0, 0);

    for (int b = 0; b < w->depth; b++) {
        struct olmo_block *blk = &w->blocks[b];

#define BLK(field, suffix, nd, s0, s1)                    \
    snprintf(name, sizeof(name), "blocks.%d." suffix, b); \
    blk->field = get(&pf, name, nd, s0, s1, 0, 0)

        BLK(ln1_w, "norm1.weight", 1, D, 0);
        BLK(ln1_b, "norm1.bias", 1, D, 0);
        load_qkv(&pf, b, D, blk);
        BLK(proj_w, "attn.proj.weight", 2, D, D);
        BLK(proj_b, "attn.proj.bias", 1, D, 0);
        BLK(ln2_w, "norm2.weight", 1, D, 0);
        BLK(ln2_b, "norm2.bias", 1, D, 0);
        BLK(fc1_w, "mlp.fc1.weight", 2, H, D);
        BLK(fc1_b, "mlp.fc1.bias", 1, H, 0);
        BLK(fc2_w, "mlp.fc2.weight", 2, D, H);
        BLK(fc2_b, "mlp.fc2.bias", 1, D, 0);
#undef BLK
    }
    w->norm_w = get(&pf, "norm.weight", 1, D, 0, 0, 0);
    w->norm_b = get(&pf, "norm.bias", 1, D, 0, 0, 0);
    pth_close(&pf);

    msg_verbose("OlmoEarth encoder: D=%d, depth=%d, heads=%d, MLP=%d", D,
                w->depth, w->heads, H);
}

void olmo_weights_free(struct olmo_weights *w)
{
    for (int s = 0; s < OLMO_NBANDSETS; s++) {
        free(w->patch_w[s]);
        free(w->patch_b[s]);
    }
    free(w->channel_embed);
    free(w->pos_embed);
    free(w->month_embed);
    for (int b = 0; b < w->depth; b++) {
        struct olmo_block *blk = &w->blocks[b];

        free(blk->ln1_w);
        free(blk->ln1_b);
        free(blk->qkv_w);
        free(blk->qkv_b);
        free(blk->proj_w);
        free(blk->proj_b);
        free(blk->ln2_w);
        free(blk->ln2_b);
        free(blk->fc1_w);
        free(blk->fc1_b);
        free(blk->fc2_w);
        free(blk->fc2_b);
    }
    free(w->norm_w);
    free(w->norm_b);
    memset(w, 0, sizeof(*w));
}
