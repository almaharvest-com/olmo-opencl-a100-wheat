/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Fine-tuning loop of the wheat segmentation head on the
 *               frozen encoder, reproducing the rslearn/Lightning recipe.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#ifndef OLMO_CL_TRAIN_H
#define OLMO_CL_TRAIN_H

#include <stdint.h>

#include "dataset.h"
#include "encoder.h"

struct train_opts {
    const char *out_dir;   /* Checkpoints and logs. */
    const char *crops;     /* Fixed val crops (data/val_test_crops.tsv). */
    const char *init_head; /* Optional initial head (2 D + 2 floats). */
    int epochs;            /* 80 */
    double lr;             /* 1e-4 */
    uint64_t seed;
    int max_train;  /* Use only the first max_train train windows (0: all). */
    int max_val;    /* Same for validation windows. */
    int augment;    /* 0: centre crops, no flips, no stochastic depth, and
                       cached train features (fast debugging runs). */
    int save_top_k; /* 3 */
};

/* Train and write, in out_dir: run_log.jsonl (one line per epoch),
 * "epoch=E-step=S.head" for the save_top_k best epochs by val wheat F1
 * and "last.head". A .head file holds W [2][D] then b [2] as float32. */
void olmo_train(struct olmo_encoder *enc, const struct olmo_dataset *ds,
                const struct olmo_norm *norm, const struct train_opts *opts);

#endif /* OLMO_CL_TRAIN_H */
