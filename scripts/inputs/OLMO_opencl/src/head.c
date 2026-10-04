/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Segmentation head with its cross-entropy loss and
 *               gradients.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"
#include "head.h"
#include "util.h"

/* Source cells and weights of torch bilinear upsampling by an integer
 * factor, align_corners=False: src = (dst + 0.5) / up - 0.5, clamped at
 * 0, second tap clamped at the last cell. */
static void bilinear_taps(int dst, int up, int n, int *i0, int *i1, double *l1)
{
    double src = (dst + 0.5) / up - 0.5;

    if (src < 0.0)
        src = 0.0;
    *i0 = (int)src;
    *i1 = *i0 + (*i0 < n - 1);
    *l1 = src - *i0;
}

double head_step(const float *W, const float *b, const float *F, int D, int g,
                 int up, const uint8_t *label, float *probs, float *dW,
                 float *db, struct head_confusion *conf)
{
    const int gg = g * g, out = g * up;
    double *L = xmalloc(sizeof(double) * HEAD_CLASSES * gg);
    double *G = xcalloc((size_t)HEAD_CLASSES * gg, sizeof(double));
    double loss = 0.0;
    long nvalid = 0;

    /* Logits at the feature resolution. */
    for (int k = 0; k < HEAD_CLASSES; k++)
        for (int p = 0; p < gg; p++) {
            double acc = b[k];

            for (int c = 0; c < D; c++)
                acc += (double)W[k * D + c] * F[(size_t)c * gg + p];
            L[k * gg + p] = acc;
        }

    for (int i = 0; i < out * out; i++)
        nvalid += label[i] != OLMO_LABEL_NODATA;

    for (int oy = 0; oy < out; oy++) {
        int y0, y1, x0, x1;
        double ly, lx;

        bilinear_taps(oy, up, g, &y0, &y1, &ly);
        for (int ox = 0; ox < out; ox++) {
            const int pix = oy * out + ox, lab = label[pix];
            double z[HEAD_CLASSES], p[HEAD_CLASSES], w[4], m, sum = 0.0;
            int cell[4], pred = 0;

            bilinear_taps(ox, up, g, &x0, &x1, &lx);
            cell[0] = y0 * g + x0;
            cell[1] = y0 * g + x1;
            cell[2] = y1 * g + x0;
            cell[3] = y1 * g + x1;
            w[0] = (1.0 - ly) * (1.0 - lx);
            w[1] = (1.0 - ly) * lx;
            w[2] = ly * (1.0 - lx);
            w[3] = ly * lx;
            for (int k = 0; k < HEAD_CLASSES; k++) {
                z[k] = 0.0;
                for (int j = 0; j < 4; j++)
                    z[k] += w[j] * L[k * gg + cell[j]];
            }
            m = z[0] > z[1] ? z[0] : z[1];
            for (int k = 0; k < HEAD_CLASSES; k++)
                sum += p[k] = exp(z[k] - m);
            for (int k = 0; k < HEAD_CLASSES; k++) {
                p[k] /= sum;
                if (probs)
                    probs[(size_t)k * out * out + pix] = (float)p[k];
                /* argmax, first maximum on ties as torch. */
                if (p[k] > p[pred])
                    pred = k;
            }
            if (lab == OLMO_LABEL_NODATA)
                continue;
            loss -= (z[lab] - m) - log(sum);
            if (conf)
                conf->conf[lab][pred]++;
            /* dloss/dz = (p - onehot) / nvalid, scattered back to the
             * feature cells with the same bilinear weights. */
            for (int k = 0; k < HEAD_CLASSES; k++) {
                const double gz = (p[k] - (k == lab)) / nvalid;

                for (int j = 0; j < 4; j++)
                    G[k * gg + cell[j]] += w[j] * gz;
            }
        }
    }

    if (dW)
        for (int k = 0; k < HEAD_CLASSES; k++)
            for (int c = 0; c < D; c++) {
                double acc = 0.0;

                for (int p = 0; p < gg; p++)
                    acc += G[k * gg + p] * F[(size_t)c * gg + p];
                dW[k * D + c] = (float)acc;
            }
    if (db)
        for (int k = 0; k < HEAD_CLASSES; k++) {
            double acc = 0.0;

            for (int p = 0; p < gg; p++)
                acc += G[k * gg + p];
            db[k] = (float)acc;
        }
    free(L);
    free(G);
    return nvalid ? loss / nvalid : 0.0;
}

void head_metrics(const struct head_confusion *c, double *precision,
                  double *recall, double *f1, double *accuracy)
{
    const double tp = c->conf[1][1], fp = c->conf[0][1], fn = c->conf[1][0];
    const double tn = c->conf[0][0], n = tp + fp + fn + tn;

    *precision = tp + fp > 0 ? tp / (tp + fp) : 0.0;
    *recall = tp + fn > 0 ? tp / (tp + fn) : 0.0;
    *f1 = 2 * tp + fp + fn > 0 ? 2 * tp / (2 * tp + fp + fn) : 0.0;
    *accuracy = n > 0 ? (tp + tn) / n : 0.0;
}
