/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      AdamW and ReduceLROnPlateau with PyTorch semantics.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#ifndef OLMO_CL_OPTIM_H
#define OLMO_CL_OPTIM_H

/* torch.optim.AdamW (amsgrad off, decoupled weight decay). */
struct adamw {
    int n;
    long t;
    double lr, beta1, beta2, eps, wd;
    double *m, *v;
};

/* Defaults of torch.optim.AdamW: betas (0.9, 0.999), eps 1e-8, weight
 * decay 0.01. */
void adamw_init(struct adamw *a, int n, double lr);
void adamw_free(struct adamw *a);

/* One step on the n parameters p with gradients g. */
void adamw_step(struct adamw *a, float *p, const float *g);

/* torch.optim.lr_scheduler.ReduceLROnPlateau, mode "min", relative
 * threshold. */
struct plateau {
    double factor, threshold, min_lr, eps;
    int patience, cooldown;
    double best;
    int num_bad, cooldown_counter;
};

void plateau_init(struct plateau *s, double factor, int patience, int cooldown,
                  double min_lr);

/* Feed the epoch metric; returns the (possibly reduced) learning rate. */
double plateau_step(struct plateau *s, double metric, double lr);

#endif /* OLMO_CL_OPTIM_H */
