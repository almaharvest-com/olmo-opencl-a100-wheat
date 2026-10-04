/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Stage-by-stage comparison with PyTorch golden dumps.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "check.h"
#include "head.h"
#include "util.h"

struct check_ctx {
    const char *dir;
    int fail;
};

/* Allowed relative error per stage: tight for the exact host-side steps,
 * looser after the 12 float32 transformer blocks. */
static double tolerance(const char *stage)
{
    if (strcmp(stage, "probs") == 0 || strcmp(stage, "loss") == 0 ||
        strcmp(stage, "dW") == 0 || strcmp(stage, "db") == 0)
        return 1e-4;
    if (strncmp(stage, "block", 5) == 0 || strcmp(stage, "norm") == 0 ||
        strcmp(stage, "features") == 0)
        return 1e-3;
    return 1e-5;
}

/* Compare got with dir/stage.f32, if that file exists: max |got - ref|
 * relative to max |ref|. */
static void compare(struct check_ctx *c, const char *stage, const float *got,
                    size_t n)
{
    char path[4096];
    struct stat st;
    double err = 0.0, scale = 0.0, tol = tolerance(stage);
    float *ref;

    xsnprintf(path, sizeof(path), "%s/%s.f32", c->dir, stage);
    if (stat(path, &st) != 0)
        return;
    ref = read_f32(path, n);
    for (size_t i = 0; i < n; i++) {
        const double e = fabs((double)got[i] - ref[i]);

        if (!(e <= err)) /* Also catches NaN. */
            err = isnan(e) ? INFINITY : e;
        if (fabs(ref[i]) > scale)
            scale = fabs(ref[i]);
    }
    err /= scale > 0.0 ? scale : 1.0;
    msg_info("  %-12s rel. error %.2e (tol %.0e)  %s", stage, err, tol,
             err <= tol ? "ok" : "FAILED");
    if (!(err <= tol))
        c->fail++;
    free(ref);
}

static void trace_cb(void *ctx, const char *stage, const float *data, size_t n)
{
    compare(ctx, stage, data, n);
}

/* Keep decisions of the recorded training-mode pass: drops.json lists
 * {"block", "branch", "keep"} in call order (attention, then MLP, per
 * block). */
static unsigned char *read_drops(const char *dir, int depth)
{
    char path[4096], *json, *p;
    unsigned char *keep = xmalloc(2 * depth);
    int n = 0;

    xsnprintf(path, sizeof(path), "%s/drops.json", dir);
    json = read_file(path);
    for (p = json; (p = strstr(p, "\"keep\"")); p++) {
        if (n == 2 * depth)
            fatal("<%s> has more than %d drop decisions", path, 2 * depth);
        keep[n++] = (unsigned char)json_bool(p, "keep", -1);
        if (keep[n - 1] > 1)
            fatal("Invalid keep value in <%s>", path);
    }
    if (n != 2 * depth)
        fatal("<%s> has %d drop decisions, expected %d", path, n, 2 * depth);
    free(json);
    return keep;
}

int olmo_check_head(const char *dir, const float *features, int D, int g)
{
    const int up = 4, out = g * up;
    struct check_ctx c = {dir, 0};
    char path[4096];
    float *F = (float *)features, *W, *b, *probs, dW[HEAD_CLASSES * 1024],
          db[HEAD_CLASSES], loss;
    uint8_t *label = xmalloc((size_t)out * out);
    FILE *fp;

    if (D > 1024)
        fatal("Embedding size %d is too large for the head check", D);
    if (!F) {
        xsnprintf(path, sizeof(path), "%s/features.f32", dir);
        F = read_f32(path, (size_t)D * g * g);
    }
    xsnprintf(path, sizeof(path), "%s/head_w.f32", dir);
    W = read_f32(path, (size_t)HEAD_CLASSES * D);
    xsnprintf(path, sizeof(path), "%s/head_b.f32", dir);
    b = read_f32(path, HEAD_CLASSES);
    xsnprintf(path, sizeof(path), "%s/label.u8", dir);
    if (!(fp = fopen(path, "rb")) ||
        fread(label, 1, (size_t)out * out, fp) != (size_t)out * out)
        fatal("Cannot read <%s>", path);
    fclose(fp);

    probs = xmalloc(sizeof(float) * HEAD_CLASSES * out * out);
    loss = (float)head_step(W, b, F, D, g, up, label, probs, dW, db, NULL);
    compare(&c, "probs", probs, (size_t)HEAD_CLASSES * out * out);
    compare(&c, "loss", &loss, 1);
    compare(&c, "dW", dW, (size_t)HEAD_CLASSES * D);
    compare(&c, "db", db, HEAD_CLASSES);

    if (F != features)
        free(F);
    free(W);
    free(b);
    free(label);
    free(probs);
    return c.fail;
}

int olmo_check(struct olmo_encoder *enc, const char *dir,
               const struct olmo_dataset *ds, const struct olmo_norm *norm)
{
    const int crop = enc->crop, D = enc->w->dim, g = enc->grid;
    const size_t nin = (size_t)enc->T * OLMO_NBANDS * crop * crop;
    struct check_ctx c = {dir, 0};
    char path[4096], *meta, *window;
    unsigned char *keep = NULL;
    float *x, *features;
    int train_mode;

    xsnprintf(path, sizeof(path), "%s/meta.json", dir);
    meta = read_file(path);
    window = json_string(meta, "window");
    train_mode = json_bool(meta, "train_mode", 0);
    msg_info("Golden <%s>: window %s, %s mode", dir, window ? window : "?",
             train_mode ? "training" : "evaluation");

    xsnprintf(path, sizeof(path), "%s/input.f32", dir);
    x = read_f32(path, nin);
    if (ds) {
        const struct olmo_window *win =
            window ? olmo_dataset_find(ds, window) : NULL;
        float *mine = xmalloc(sizeof(float) * nin);

        if (!win)
            fatal("Window <%s> of <%s> is not in the dataset", window, dir);
        olmo_make_crop(win, norm, (int)json_long(meta, "offset_x", -1),
                       (int)json_long(meta, "offset_y", -1), crop, 0, 0, mine,
                       NULL);
        compare(&c, "input", mine, nin);
        free(x);
        x = mine; /* Continue from the C data path. */
    }
    if (train_mode)
        keep = read_drops(dir, enc->w->depth);

    features = xmalloc(sizeof(float) * D * g * g);
    encoder_forward(enc, x, keep, features, trace_cb, &c);
    compare(&c, "features", features, (size_t)D * g * g);
    c.fail += olmo_check_head(dir, features, D, g);

    free(features);
    free(keep);
    free(x);
    free(window);
    free(meta);
    return c.fail;
}
