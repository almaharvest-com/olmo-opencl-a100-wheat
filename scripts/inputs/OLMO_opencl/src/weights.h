/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Frozen OlmoEarth encoder weights for Sentinel-2 L2A, read
 *               from the Hugging Face weights.pth state_dict.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#ifndef OLMO_CL_WEIGHTS_H
#define OLMO_CL_WEIGHTS_H

/* Sentinel-2 L2A band sets, in OlmoEarth's band order
 * B02 B03 B04 B08 | B05 B06 B07 B8A B11 B12 | B01 B09. */
#define OLMO_NBANDSETS  3
#define OLMO_NBANDS     12
#define OLMO_BASE_PATCH 8 /* Patch-embed kernel size of the checkpoint. */
#define OLMO_MAX_DEPTH  24

extern const int olmo_bandset_first[OLMO_NBANDSETS];
extern const int olmo_bandset_size[OLMO_NBANDSETS];

/* All arrays are C-contiguous float32 in PyTorch layout ([out, in] for
 * linear layers, [out, in, kh, kw] for convolutions). */
struct olmo_block {
    float *ln1_w, *ln1_b;
    float *qkv_w, *qkv_b; /* [3D, D] and [3D]: q, k, v stacked by row. */
    float *proj_w, *proj_b;
    float *ln2_w, *ln2_b;
    float *fc1_w, *fc1_b; /* [H, D] */
    float *fc2_w, *fc2_b; /* [D, H] */
};

struct olmo_weights {
    int dim;                        /* D, embedding size (192 for Tiny). */
    int hidden;                     /* H, MLP hidden size. */
    int depth;                      /* Number of transformer blocks. */
    int heads;                      /* Attention heads. */
    int enc_dim;                    /* D / 4, width of each encoding slot. */
    int max_seqlen;                 /* Rows of the time and month tables. */
    float *patch_w[OLMO_NBANDSETS]; /* [D, |s|, 8, 8] */
    float *patch_b[OLMO_NBANDSETS];
    float *channel_embed; /* [3, D/4] */
    float *pos_embed;     /* [max_seqlen, D/4], time position. */
    float *month_embed;   /* [12, D/4] */
    struct olmo_block blocks[OLMO_MAX_DEPTH];
    float *norm_w, *norm_b;
};

/* Load the encoder from a state_dict. heads is not stored in the
 * weights and comes from the model's config.json (3 for Tiny). Fatal on
 * any missing tensor or shape mismatch. */
void olmo_weights_load(struct olmo_weights *w, const char *path, int heads);

void olmo_weights_free(struct olmo_weights *w);

#endif /* OLMO_CL_WEIGHTS_H */
