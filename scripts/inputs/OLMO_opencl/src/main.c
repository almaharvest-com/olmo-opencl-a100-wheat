/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Command line entry point: fine-tune the ALMA wheat
 *               segmentation head on a frozen OlmoEarth encoder in OpenCL.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "check.h"
#include "dataset.h"
#include "encoder.h"
#include "ocl_backend.h"
#include "ops.h"
#include "pth_loader.h"
#include "selftest.h"
#include "train.h"
#include "util.h"
#include "weights.h"

static void usage(void)
{
    fputs(
        "Usage: olmo_cl COMMAND [key=value ...]\n"
        "\n"
        "Commands:\n"
        "  info weights=FILE      list the tensors of a PyTorch state_dict\n"
        "  weights weights=FILE [heads=3]\n"
        "                         load and check the S2 encoder weights\n"
        "  data dataset=DIR [group=random_split] [order=readdir|index]\n"
        "                         load and summarise the rslearn windows\n"
        "  selftest [device=auto|gpu|cpu] [platform=NAME] [tokens=512]\n"
        "                         check the OpenCL kernels against C loops\n"
        "  check golden=DIR[,DIR...] weights=FILE [dataset=DIR]\n"
        "        [norm=data/s2_norm.tsv] [device=...] [platform=NAME]\n"
        "                         compare the encoder with PyTorch dumps\n"
        "  headcheck golden=DIR[,DIR...] [dim=192]\n"
        "                         check the head on the golden features\n"
        "  train dataset=DIR weights=FILE out=DIR [epochs=80] [lr=1e-4]\n"
        "        [seed=42] [norm=data/s2_norm.tsv]\n"
        "        [crops=data/val_test_crops.tsv] [init=HEAD] [top_k=3]\n"
        "        [augment=1] [max_train=0] [max_val=0] [overwrite=0]\n"
        "        [order=readdir|index] [device=...] [platform=NAME]\n"
        "                         fine-tune the wheat segmentation head\n"
        "\n"
        "Environment: OLMO_CL_VERBOSE=0..3, OLMO_CL_PROFILE=1 (kernel times).\n"
        "rusticl exposes AMD GPUs only with RUSTICL_ENABLE=radeonsi.\n",
        stderr);
    exit(EXIT_FAILURE);
}

/* Return the value of key=value among the arguments, or def. */
static const char *arg(int argc, char **argv, const char *key, const char *def)
{
    size_t n = strlen(key);

    for (int i = 2; i < argc; i++)
        if (strncmp(argv[i], key, n) == 0 && argv[i][n] == '=')
            return argv[i] + n + 1;
    return def;
}

static int cmd_info(int argc, char **argv)
{
    const char *path = arg(argc, argv, "weights", NULL);
    static const char *dtypes[] = {"f32", "f16", "bf16"};
    struct pth_file pf;
    long total = 0;

    if (!path)
        fatal("info: weights= is required");
    pth_open(&pf, path);
    for (int i = 0; i < pf.ntensors; i++) {
        const struct pth_tensor *t = &pf.tensors[i];

        printf("%s\t%s\t[", t->name, dtypes[t->dtype]);
        for (int d = 0; d < t->ndim; d++)
            printf(d ? ",%ld" : "%ld", t->shape[d]);
        printf("]\n");
        total += pth_numel(t);
    }
    msg_info("%d tensors, %ld parameters", pf.ntensors, total);
    pth_close(&pf);
    return 0;
}

static int cmd_weights(int argc, char **argv)
{
    const char *path = arg(argc, argv, "weights", NULL);
    struct olmo_weights w;
    long n = 0;

    if (!path)
        fatal("weights: weights= is required");
    olmo_weights_load(&w, path, atoi(arg(argc, argv, "heads", "3")));
    for (int s = 0; s < OLMO_NBANDSETS; s++)
        n += (long)w.dim *
             (olmo_bandset_size[s] * OLMO_BASE_PATCH * OLMO_BASE_PATCH + 1);
    n += (long)w.depth * (4L * w.dim + 4L * w.dim * w.dim + 4L * w.dim +
                          2L * w.dim * w.hidden + w.hidden + w.dim);
    n += 2L * w.dim + OLMO_NBANDSETS * w.enc_dim;
    printf("dim=%d depth=%d heads=%d hidden=%d enc_dim=%d max_seqlen=%d "
           "encoder_params_used=%ld\n",
           w.dim, w.depth, w.heads, w.hidden, w.enc_dim, w.max_seqlen, n);
    olmo_weights_free(&w);
    return 0;
}

static int cmd_data(int argc, char **argv)
{
    const char *dir = arg(argc, argv, "dataset", NULL);
    const char *order = arg(argc, argv, "order", "readdir");
    struct olmo_dataset ds;
    int count[4] = {0}, wmin = 1 << 30, hmin = 1 << 30, wmax = 0, hmax = 0;
    long cls[3] = {0};

    if (!dir)
        fatal("data: dataset= is required");
    if (strcmp(order, "readdir") && strcmp(order, "index"))
        fatal("data: order=%s, expected readdir or index", order);
    olmo_dataset_load(&ds, dir, arg(argc, argv, "group", "random_split"),
                      "sentinel2", "label",
                      strcmp(order, "index") ? ORDER_READDIR : ORDER_INDEX);
    for (int i = 0; i < ds.nwindows; i++) {
        const struct olmo_window *win = &ds.windows[i];
        size_t plane = (size_t)win->width * win->height;

        count[win->split]++;
        wmin = win->width < wmin ? win->width : wmin;
        hmin = win->height < hmin ? win->height : hmin;
        wmax = win->width > wmax ? win->width : wmax;
        hmax = win->height > hmax ? win->height : hmax;
        for (size_t p = 0; p < plane; p++)
            cls[win->label[p] == OLMO_LABEL_NODATA ? 2 : win->label[p]]++;
        /* Report the timestep order once, and any window that differs. */
        for (int t = 0; t < win->ntime; t++)
            if (i == 0 || strcmp(win->layer[t], ds.windows[0].layer[t])) {
                printf("timestep order (%s):", win->name);
                for (int u = 0; u < win->ntime; u++)
                    printf(" %s", win->layer[u]);
                printf("\n");
                break;
            }
    }
    printf("windows=%d train=%d val=%d test=%d other=%d timesteps=%d\n",
           ds.nwindows, count[SPLIT_TRAIN], count[SPLIT_VAL], count[SPLIT_TEST],
           count[SPLIT_OTHER], ds.ntime);
    printf("size: %d-%d x %d-%d px\n", wmin, wmax, hmin, hmax);
    printf("label pixels: background=%ld wheat=%ld nodata=%ld\n", cls[0],
           cls[1], cls[2]);
    if (wmin < 64 || hmin < 64)
        msg_warning("Some windows are smaller than the 64 px training crop");
    olmo_dataset_free(&ds);
    return 0;
}

static int cmd_selftest(int argc, char **argv)
{
    struct ocl_backend ocl;
    struct olmo_ops o;
    int fail;

    ocl_init(&ocl, arg(argc, argv, "device", "auto"),
             arg(argc, argv, "platform", NULL));
    ops_init(&o, &ocl);
    fail = olmo_selftest(&o, atoi(arg(argc, argv, "tokens", "512")));
    if (ocl.profile)
        ops_profile_report();
    ops_free(&o);
    ocl_free(&ocl);
    return fail ? EXIT_FAILURE : 0;
}

static int cmd_headcheck(int argc, char **argv)
{
    const char *golden = arg(argc, argv, "golden", NULL);
    const int D = atoi(arg(argc, argv, "dim", "192"));
    char *list, *dir, *save;
    int fail = 0;

    if (!golden)
        fatal("headcheck: golden= is required");
    list = xstrdup(golden);
    for (dir = strtok_r(list, ",", &save); dir;
         dir = strtok_r(NULL, ",", &save)) {
        msg_info("Golden <%s>:", dir);
        fail += olmo_check_head(dir, NULL, D, 16);
    }
    free(list);
    return fail ? EXIT_FAILURE : 0;
}

static int cmd_train(int argc, char **argv)
{
    const char *dsdir = arg(argc, argv, "dataset", NULL);
    const char *wpath = arg(argc, argv, "weights", NULL);
    const char *order = arg(argc, argv, "order", "readdir");
    struct train_opts opts = {
        .out_dir = arg(argc, argv, "out", NULL),
        .crops = arg(argc, argv, "crops", "data/val_test_crops.tsv"),
        .init_head = arg(argc, argv, "init", NULL),
        .epochs = atoi(arg(argc, argv, "epochs", "80")),
        .lr = atof(arg(argc, argv, "lr", "1e-4")),
        .seed = strtoull(arg(argc, argv, "seed", "42"), NULL, 10),
        .max_train = atoi(arg(argc, argv, "max_train", "0")),
        .max_val = atoi(arg(argc, argv, "max_val", "0")),
        .augment = atoi(arg(argc, argv, "augment", "1")),
        .save_top_k = atoi(arg(argc, argv, "top_k", "3")),
    };
    struct olmo_dataset ds;
    struct olmo_norm norm;
    struct olmo_weights w;
    struct ocl_backend ocl;
    struct olmo_ops o;
    struct olmo_encoder enc;
    char path[4096];
    struct stat st;
    FILE *fp;

    if (!dsdir || !wpath || !opts.out_dir)
        fatal("train: dataset=, weights= and out= are required");
    if (strcmp(order, "readdir") && strcmp(order, "index"))
        fatal("train: order=%s, expected readdir or index", order);
    if (opts.epochs < 1)
        fatal("train: epochs must be positive");
    xsnprintf(path, sizeof(path), "%s/run_log.jsonl", opts.out_dir);
    if (stat(path, &st) == 0 && !atoi(arg(argc, argv, "overwrite", "0")))
        fatal("<%s> exists: choose another out= or pass overwrite=1",
              opts.out_dir);
    if (mkdir(opts.out_dir, 0775) != 0 && stat(opts.out_dir, &st) != 0)
        fatal("Cannot create <%s>", opts.out_dir);

    olmo_dataset_load(&ds, dsdir, arg(argc, argv, "group", "random_split"),
                      "sentinel2", "label",
                      strcmp(order, "index") ? ORDER_READDIR : ORDER_INDEX);
    olmo_norm_load(&norm, arg(argc, argv, "norm", "data/s2_norm.tsv"));
    olmo_weights_load(&w, wpath, atoi(arg(argc, argv, "heads", "3")));
    ocl_init(&ocl, arg(argc, argv, "device", "auto"),
             arg(argc, argv, "platform", NULL));
    ops_init(&o, &ocl);
    encoder_init(&enc, &o, &w, ds.ntime, 64, 4);

    /* Everything needed to interpret or repeat the run. */
    xsnprintf(path, sizeof(path), "%s/run_config.json", opts.out_dir);
    if (!(fp = fopen(path, "w")))
        fatal("Cannot write <%s>", path);
    fprintf(fp, "{\n  \"command\": \"");
    for (int i = 0; i < argc; i++)
        fprintf(fp, i ? " %s" : "%s", argv[i]);
    fprintf(fp, "\",\n  \"device\": \"%s\",\n  \"platform\": \"%s\",\n",
            ocl.device_name, ocl.platform_name);
    fprintf(fp, "  \"timestep_order\": [");
    for (int t = 0; t < ds.ntime; t++)
        fprintf(fp, t ? ", \"%s\"" : "\"%s\"", ds.windows[0].layer[t]);
    fprintf(fp,
            "],\n  \"dim\": %d,\n  \"head_file\": \"W[2][%d] then "
            "b[2], float32\"\n}\n",
            w.dim, w.dim);
    fclose(fp);

    olmo_train(&enc, &ds, &norm, &opts);
    if (ocl.profile)
        ops_profile_report();

    encoder_free(&enc);
    ops_free(&o);
    ocl_free(&ocl);
    olmo_weights_free(&w);
    olmo_dataset_free(&ds);
    return 0;
}

static int cmd_check(int argc, char **argv)
{
    const char *golden = arg(argc, argv, "golden", NULL);
    const char *wpath = arg(argc, argv, "weights", NULL);
    const char *dsdir = arg(argc, argv, "dataset", NULL);
    const char *npath = arg(argc, argv, "norm", "data/s2_norm.tsv");
    struct olmo_dataset ds;
    struct olmo_norm norm;
    struct olmo_weights w;
    struct ocl_backend ocl;
    struct olmo_ops o;
    struct olmo_encoder enc;
    char *list, *dir, *save;
    int fail = 0, T = 6;

    if (!golden || !wpath)
        fatal("check: golden= and weights= are required");
    if (dsdir) {
        olmo_dataset_load(&ds, dsdir, arg(argc, argv, "group", "random_split"),
                          "sentinel2", "label", ORDER_READDIR);
        olmo_norm_load(&norm, npath);
        T = ds.ntime;
    }
    olmo_weights_load(&w, wpath, atoi(arg(argc, argv, "heads", "3")));
    ocl_init(&ocl, arg(argc, argv, "device", "auto"),
             arg(argc, argv, "platform", NULL));
    ops_init(&o, &ocl);
    encoder_init(&enc, &o, &w, T, 64, 4);

    list = xstrdup(golden);
    for (dir = strtok_r(list, ",", &save); dir;
         dir = strtok_r(NULL, ",", &save))
        fail += olmo_check(&enc, dir, dsdir ? &ds : NULL, &norm);
    free(list);
    if (ocl.profile)
        ops_profile_report();
    msg_info(fail ? "%d stage(s) FAILED" : "All stages within tolerance", fail);

    encoder_free(&enc);
    ops_free(&o);
    ocl_free(&ocl);
    olmo_weights_free(&w);
    if (dsdir)
        olmo_dataset_free(&ds);
    return fail ? EXIT_FAILURE : 0;
}

int main(int argc, char **argv)
{
    const char *v = getenv("OLMO_CL_VERBOSE");

    if (v)
        msg_level = atoi(v);
    if (argc < 2)
        usage();
    if (strcmp(argv[1], "info") == 0)
        return cmd_info(argc, argv);
    if (strcmp(argv[1], "weights") == 0)
        return cmd_weights(argc, argv);
    if (strcmp(argv[1], "data") == 0)
        return cmd_data(argc, argv);
    if (strcmp(argv[1], "train") == 0)
        return cmd_train(argc, argv);
    if (strcmp(argv[1], "headcheck") == 0)
        return cmd_headcheck(argc, argv);
    if (strcmp(argv[1], "check") == 0)
        return cmd_check(argc, argv);
    if (strcmp(argv[1], "selftest") == 0)
        return cmd_selftest(argc, argv);
    usage();
    return EXIT_FAILURE;
}
