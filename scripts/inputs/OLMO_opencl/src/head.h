/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Segmentation head (bilinear upsampling, 1x1 convolution,
 *               softmax) with its cross-entropy loss and gradients.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#ifndef OLMO_CL_HEAD_H
#define OLMO_CL_HEAD_H

#include <stdint.h>

#define HEAD_CLASSES 2

/* Pixel counts over valid pixels: conf[label][prediction]. */
struct head_confusion {
    long conf[HEAD_CLASSES][HEAD_CLASSES];
};

/* rslearn's decoder Upsample(bilinear, x up) -> Conv 1x1 (D -> 2) ->
 * softmax, evaluated as conv then upsampling (the same, both being
 * linear and the bilinear weights summing to 1).
 *
 * F is [D][g][g], W [2][D], b [2]; label [g up][g up] holds 0, 1 or
 * nodata (255, excluded). Returns the cross-entropy averaged over valid
 * pixels (0 if there is none). probs [2][g up][g up], dW, db and conf may
 * each be NULL; the gradients are overwritten, the counts accumulated. */
double head_step(const float *W, const float *b, const float *F, int D, int g,
                 int up, const uint8_t *label, float *probs, float *dW,
                 float *db, struct head_confusion *conf);

/* Class-1 (wheat) precision, recall and F1, and overall accuracy, as
 * torchmetrics computes them (0 when undefined). */
void head_metrics(const struct head_confusion *c, double *precision,
                  double *recall, double *f1, double *accuracy);

#endif /* OLMO_CL_HEAD_H */
