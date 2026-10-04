/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Fine-tuning loop of the wheat segmentation head on the
 *               frozen encoder.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "head.h"
#include "optim.h"
#include "train.h"
#include "util.h"

#define CROP 64
#define UP   4

/* PCG32 (O'Neill), for reproducible crops, flips, shuffles and
 * stochastic depth. */
struct pcg {
    uint64_t state, inc;
};

static uint32_t pcg_next(struct pcg *r)
{
    uint64_t old = r->state;
    uint32_t xs, rot;

    r->state = old * 6364136223846793005ULL + r->inc;
    xs = (uint32_t)(((old >> 18) ^ old) >> 27);
    rot = (uint32_t)(old >> 59);
    return (xs >> rot) | (xs << ((-rot) & 31));
}

static void pcg_seed(struct pcg *r, uint64_t seed)
{
    r->state = 0;
    r->inc = (seed << 1) | 1;
    pcg_next(r);
    r->state += seed;
    pcg_next(r);
}

/* Uniform integer in [0, n), without modulo bias. */
static uint32_t pcg_below(struct pcg *r, uint32_t n)
{
    uint32_t threshold = -n % n, x;

    do
        x = pcg_next(r);
    while (x < threshold);
    return x % n;
}

static double pcg_uniform(struct pcg *r)
{
    return pcg_next(r) / 4294967296.0;
}

/* One training or validation example. */
struct example {
    const struct olmo_window *win;
    int ox, oy;
    float *features; /* Cached encoder output, or NULL. */
};

/* Fixed val crop offsets of rslearn (fix_patch_pick), by window name. */
static int find_crop(const char *path, const char *name, int *ox, int *oy)
{
    FILE *fp = fopen(path, "r");
    char line[512], wname[400], split[16];
    int found = 0;

    if (!fp)
        fatal("Cannot open crop table <%s>", path);
    while (!found && fgets(line, sizeof(line), fp))
        if (line[0] != '#' &&
            sscanf(line, "%399s %15s %d %d", wname, split, ox, oy) == 4 &&
            strcmp(wname, name) == 0)
            found = 1;
    fclose(fp);
    return found;
}

static void write_head(const char *path, const float *W, const float *b, int D)
{
    FILE *fp = fopen(path, "wb");

    if (!fp || fwrite(W, sizeof(float), (size_t)2 * D, fp) != (size_t)2 * D ||
        fwrite(b, sizeof(float), 2, fp) != 2 || fclose(fp) != 0)
        fatal("Cannot write <%s>", path);
}

static void read_head(const char *path, float *W, float *b, int D)
{
    FILE *fp = fopen(path, "rb");

    if (!fp || fread(W, sizeof(float), (size_t)2 * D, fp) != (size_t)2 * D ||
        fread(b, sizeof(float), 2, fp) != 2 || fgetc(fp) != EOF)
        fatal("<%s> is not a head file of %d floats", path, 2 * D + 2);
    fclose(fp);
}

/* ModelCheckpoint(monitor=val F1, mode=max, save_top_k): keep the k best
 * epochs; a new score must strictly beat the worst kept one. */
struct topk {
    int k, n;
    double score[16];
    char path[16][4096];
};

static void topk_offer(struct topk *t, double score, const char *path,
                       const float *W, const float *b, int D)
{
    int worst = 0;

    if (isnan(score))
        return;
    if (t->n < t->k) {
        write_head(path, W, b, D);
        t->score[t->n] = score;
        strcpy(t->path[t->n++], path);
        return;
    }
    for (int i = 1; i < t->n; i++)
        if (t->score[i] < t->score[worst])
            worst = i;
    if (!(score > t->score[worst]))
        return;
    if (remove(t->path[worst]) != 0)
        msg_warning("Cannot remove <%s>", t->path[worst]);
    write_head(path, W, b, D);
    t->score[worst] = score;
    strcpy(t->path[worst], path);
}

static double now(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void olmo_train(struct olmo_encoder *enc, const struct olmo_dataset *ds,
                const struct olmo_norm *norm, const struct train_opts *opts)
{
    const int D = enc->w->dim, g = enc->grid, depth = enc->w->depth;
    const size_t nfeat = (size_t)D * g * g;
    const size_t nin = (size_t)enc->T * OLMO_NBANDS * CROP * CROP;
    struct example *train = xcalloc(ds->nwindows, sizeof(*train));
    struct example *val = xcalloc(ds->nwindows, sizeof(*val));
    float *x = xmalloc(sizeof(float) * nin),
          *F = xmalloc(sizeof(float) * nfeat);
    float W[2 * 1024], b[2], dW[2 * 1024 + 2], params[2 * 1024 + 2];
    uint8_t label[CROP * CROP];
    unsigned char keep[2 * OLMO_MAX_DEPTH];
    struct topk top = {0};
    struct adamw opt;
    struct plateau sched;
    struct pcg rng;
    char path[4096];
    FILE *log;
    int ntrain = 0, nval = 0;
    long step = 0;
    double lr = opts->lr;

    if (D > 1024)
        fatal("Embedding size %d is too large", D);
    if (opts->save_top_k < 1 || opts->save_top_k > 16)
        fatal("save_top_k must be within 1..16");
    top.k = opts->save_top_k;

    /* Splits, in window-name order. */
    for (int i = 0; i < ds->nwindows; i++) {
        const struct olmo_window *win = &ds->windows[i];

        if (win->width < CROP || win->height < CROP)
            fatal("Window <%s> is smaller than the %d px crop", win->name,
                  CROP);
        if (win->split == SPLIT_TRAIN &&
            (!opts->max_train || ntrain < opts->max_train))
            train[ntrain++].win = win;
        else if (win->split == SPLIT_VAL &&
                 (!opts->max_val || nval < opts->max_val)) {
            val[nval].win = win;
            if (!find_crop(opts->crops, win->name, &val[nval].ox,
                           &val[nval].oy))
                fatal("No fixed crop for val window <%s> in <%s>", win->name,
                      opts->crops);
            nval++;
        }
    }
    if (!ntrain || !nval)
        fatal("Need train and val windows, found %d and %d", ntrain, nval);

    /* Head initialisation as torch.nn.Conv2d: U(+-1 / sqrt(fan_in)). */
    pcg_seed(&rng, opts->seed);
    if (opts->init_head)
        read_head(opts->init_head, W, b, D);
    else {
        const double bound = 1.0 / sqrt((double)D);

        for (int i = 0; i < 2 * D; i++)
            W[i] = (float)((2.0 * pcg_uniform(&rng) - 1.0) * bound);
        for (int i = 0; i < 2; i++)
            b[i] = (float)((2.0 * pcg_uniform(&rng) - 1.0) * bound);
    }
    adamw_init(&opt, 2 * D + 2, lr);
    plateau_init(&sched, 0.2, 2, 10, 0.0);

    xsnprintf(path, sizeof(path), "%s/run_log.jsonl", opts->out_dir);
    if (!(log = fopen(path, "w")))
        fatal("Cannot write <%s>", path);

    /* The encoder is frozen and the val crops fixed: encode them once. */
    msg_info("Encoding %d validation crops...", nval);
    for (int i = 0; i < nval; i++) {
        val[i].features = xmalloc(sizeof(float) * nfeat);
        olmo_make_crop(val[i].win, norm, val[i].ox, val[i].oy, CROP, 0, 0, x,
                       NULL);
        encoder_forward(enc, x, NULL, val[i].features, NULL, NULL);
    }
    if (!opts->augment)
        for (int i = 0; i < ntrain; i++) {
            train[i].ox = (train[i].win->width - CROP) / 2;
            train[i].oy = (train[i].win->height - CROP) / 2;
        }

    msg_info("Training on %d windows, validating on %d, %d epochs%s", ntrain,
             nval, opts->epochs,
             opts->augment ? "" : " (no augmentation, cached features)");
    for (int epoch = 0; epoch < opts->epochs; epoch++) {
        struct head_confusion conf = {0};
        double t0 = now(), train_loss = 0.0, val_loss = 0.0, prec, rec, f1, acc;

        /* Shuffle (Fisher-Yates), as the train DataLoader. */
        for (int i = ntrain - 1; i > 0; i--) {
            const int j = (int)pcg_below(&rng, (uint32_t)i + 1);
            struct example tmp = train[i];

            train[i] = train[j];
            train[j] = tmp;
        }
        for (int i = 0; i < ntrain; i++) {
            struct example *ex = &train[i];
            const float *feat = F;
            int flip_h = 0, flip_v = 0;

            if (opts->augment) {
                /* rslearn: uniform patch offset, then Flip with two
                 * independent coin tosses. */
                ex->ox = (int)pcg_below(&rng, ex->win->width - CROP + 1);
                ex->oy = (int)pcg_below(&rng, ex->win->height - CROP + 1);
                flip_h = pcg_next(&rng) & 1;
                flip_v = pcg_next(&rng) & 1;
                for (int k = 0; k < 2 * depth; k++)
                    keep[k] = pcg_uniform(&rng) >= OLMO_DROP_PATH;
            }
            olmo_make_crop(ex->win, norm, ex->ox, ex->oy, CROP, flip_h, flip_v,
                           x, label);
            if (!opts->augment && ex->features)
                feat = ex->features;
            else {
                encoder_forward(enc, x, opts->augment ? keep : NULL, F, NULL,
                                NULL);
                if (!opts->augment) {
                    ex->features = xmalloc(sizeof(float) * nfeat);
                    memcpy(ex->features, F, sizeof(float) * nfeat);
                }
            }
            train_loss += head_step(W, b, feat, D, g, UP, label, NULL, dW,
                                    dW + 2 * D, NULL);
            memcpy(params, W, sizeof(float) * 2 * D);
            memcpy(params + 2 * D, b, sizeof(float) * 2);
            opt.lr = lr;
            adamw_step(&opt, params, dW);
            memcpy(W, params, sizeof(float) * 2 * D);
            memcpy(b, params + 2 * D, sizeof(float) * 2);
            step++;
        }
        train_loss /= ntrain;

        for (int i = 0; i < nval; i++) {
            olmo_make_crop(val[i].win, norm, val[i].ox, val[i].oy, CROP, 0, 0,
                           x, label);
            val_loss += head_step(W, b, val[i].features, D, g, UP, label, NULL,
                                  NULL, NULL, &conf);
        }
        val_loss /= nval;
        head_metrics(&conf, &prec, &rec, &f1, &acc);

        msg_info("epoch %3d  step %6ld  train_loss %.5f  val_loss %.5f  "
                 "val_f1 %.4f  P %.4f  R %.4f  lr %.2e  %.1f s",
                 epoch, step, train_loss, val_loss, f1, prec, rec, lr,
                 now() - t0);
        fprintf(log,
                "{\"epoch\": %d, \"step\": %ld, \"train_loss\": %.8g, "
                "\"val_loss\": %.8g, \"val_wheat_f1\": %.8g, "
                "\"val_wheat_precision\": %.8g, \"val_wheat_recall\": %.8g, "
                "\"val_accuracy\": %.8g, \"lr\": %.8g, \"seconds\": %.3f}\n",
                epoch, step, train_loss, val_loss, f1, prec, rec, acc, lr,
                now() - t0);
        fflush(log);

        /* Lightning file names: 0-based epoch, global step after it. */
        xsnprintf(path, sizeof(path), "%s/epoch=%d-step=%ld.head",
                  opts->out_dir, epoch, step);
        topk_offer(&top, f1, path, W, b, D);
        xsnprintf(path, sizeof(path), "%s/last.head", opts->out_dir);
        write_head(path, W, b, D);

        /* ReduceLROnPlateau on the epoch train loss. */
        lr = plateau_step(&sched, train_loss, lr);
    }
    fclose(log);

    for (int i = 0; i < top.n; i++)
        msg_info("Kept %s (val F1 %.4f)", top.path[i], top.score[i]);
    for (int i = 0; i < ntrain; i++)
        free(train[i].features);
    for (int i = 0; i < nval; i++)
        free(val[i].features);
    free(train);
    free(val);
    free(x);
    free(F);
    adamw_free(&opt);
}
