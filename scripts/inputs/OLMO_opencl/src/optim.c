/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      AdamW and ReduceLROnPlateau with PyTorch semantics.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "optim.h"
#include "util.h"

void adamw_init(struct adamw *a, int n, double lr)
{
    memset(a, 0, sizeof(*a));
    a->n = n;
    a->lr = lr;
    a->beta1 = 0.9;
    a->beta2 = 0.999;
    a->eps = 1e-8;
    a->wd = 0.01;
    a->m = xcalloc(n, sizeof(double));
    a->v = xcalloc(n, sizeof(double));
}

void adamw_free(struct adamw *a)
{
    free(a->m);
    free(a->v);
    memset(a, 0, sizeof(*a));
}

void adamw_step(struct adamw *a, float *p, const float *g)
{
    double bc1, bc2, step;

    a->t++;
    bc1 = 1.0 - pow(a->beta1, (double)a->t);
    bc2 = 1.0 - pow(a->beta2, (double)a->t);
    step = a->lr / bc1;
    for (int i = 0; i < a->n; i++) {
        double x = p[i];

        x *= 1.0 - a->lr * a->wd;
        a->m[i] = a->beta1 * a->m[i] + (1.0 - a->beta1) * g[i];
        a->v[i] = a->beta2 * a->v[i] + (1.0 - a->beta2) * g[i] * g[i];
        x -= step * a->m[i] / (sqrt(a->v[i]) / sqrt(bc2) + a->eps);
        p[i] = (float)x;
    }
}

void plateau_init(struct plateau *s, double factor, int patience, int cooldown,
                  double min_lr)
{
    memset(s, 0, sizeof(*s));
    s->factor = factor;
    s->patience = patience;
    s->cooldown = cooldown;
    s->min_lr = min_lr;
    s->threshold = 1e-4;
    s->eps = 1e-8;
    s->best = INFINITY;
}

double plateau_step(struct plateau *s, double metric, double lr)
{
    if (metric < s->best * (1.0 - s->threshold)) {
        s->best = metric;
        s->num_bad = 0;
    }
    else
        s->num_bad++;
    if (s->cooldown_counter > 0) {
        s->cooldown_counter--;
        s->num_bad = 0;
    }
    if (s->num_bad > s->patience) {
        double new_lr = fmax(lr * s->factor, s->min_lr);

        if (lr - new_lr > s->eps)
            lr = new_lr;
        s->cooldown_counter = s->cooldown;
        s->num_bad = 0;
    }
    return lr;
}
