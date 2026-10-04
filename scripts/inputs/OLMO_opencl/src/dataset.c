/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      rslearn window dataset: Sentinel-2 time series and label
 *               rasters, read with GDAL.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <ctype.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <gdal.h>

#include "dataset.h"
#include "util.h"

/* S2 band directory and file written by rslearn for the configured band
 * set. */
#define S2_BANDS_DIR "B02_B03_B04_B08_B05_B06_B07_B8A_B11_B12_B01_B09"

const char *olmo_split_name(enum olmo_split s)
{
    static const char *names[] = {"train", "val", "test", "other"};

    return names[s];
}

static int exists(const char *path)
{
    struct stat st;

    return stat(path, &st) == 0;
}

static enum olmo_split parse_split(const char *s)
{
    if (!s)
        return SPLIT_OTHER;
    if (strcmp(s, "train") == 0)
        return SPLIT_TRAIN;
    if (strcmp(s, "val") == 0)
        return SPLIT_VAL;
    if (strcmp(s, "test") == 0)
        return SPLIT_TEST;
    return SPLIT_OTHER;
}

/* Item group index of a layer directory name: "sentinel2" is 0,
 * "sentinel2.3" is 3; -1 if the name is not a group of layer. */
static int group_index(const char *dname, const char *layer)
{
    size_t n = strlen(layer);
    char *end;
    long idx;

    if (strncmp(dname, layer, n) != 0)
        return -1;
    if (dname[n] == '\0')
        return 0;
    if (dname[n] != '.' || !isdigit((unsigned char)dname[n + 1]))
        return -1;
    idx = strtol(dname + n + 1, &end, 10);
    return *end ? -1 : (int)idx;
}

static int cmp_group(const void *a, const void *b)
{
    const char *const *sa = a, *const *sb = b;
    const char *da = strrchr(*sa, '.'), *db = strrchr(*sb, '.');
    int ia = da ? atoi(da + 1) : 0, ib = db ? atoi(db + 1) : 0;

    return (ia > ib) - (ia < ib);
}

/* List the completed item groups of layer in the window, in the order
 * requested. */
static int list_layers(struct olmo_window *win, const char *wdir,
                       const char *layer, enum olmo_layer_order order)
{
    char path[4096];
    struct dirent *de;
    DIR *d;
    int n = 0;

    xsnprintf(path, sizeof(path), "%s/layers", wdir);
    d = opendir(path);
    if (!d)
        fatal("Cannot open <%s>", path);
    /* No sorting: readdir() is the order Python's Path.iterdir() sees. */
    while ((de = readdir(d))) {
        char done[4096];

        if (group_index(de->d_name, layer) < 0)
            continue;
        xsnprintf(done, sizeof(done), "%s/%s/completed", path, de->d_name);
        if (!exists(done))
            continue;
        if (n == OLMO_MAX_TIMESTEPS)
            fatal("Window <%s> has more than %d <%s> layers", win->name,
                  OLMO_MAX_TIMESTEPS, layer);
        win->layer[n++] = xstrdup(de->d_name);
    }
    closedir(d);
    if (order == ORDER_INDEX)
        qsort(win->layer, n, sizeof(char *), cmp_group);
    return n;
}

/* Read a whole GeoTIFF into buf as type, checking its size and band
 * count. */
static void read_raster(const char *path, int width, int height, int nbands,
                        GDALDataType type, void *buf)
{
    GDALDatasetH ds = GDALOpen(path, GA_ReadOnly);
    CPLErr err;

    if (!ds)
        fatal("Cannot open raster <%s>", path);
    if (GDALGetRasterXSize(ds) != width || GDALGetRasterYSize(ds) != height)
        fatal("Raster <%s> is %dx%d, expected %dx%d from the window bounds",
              path, GDALGetRasterXSize(ds), GDALGetRasterYSize(ds), width,
              height);
    if (GDALGetRasterCount(ds) != nbands)
        fatal("Raster <%s> has %d bands, expected %d", path,
              GDALGetRasterCount(ds), nbands);
    /* Band-sequential: band b lands at buf + b * width * height. */
    err = GDALDatasetRasterIO(ds, GF_Read, 0, 0, width, height, buf, width,
                              height, type, nbands, NULL, 0, 0, 0);
    if (err != CE_None)
        fatal("Reading raster <%s> failed", path);
    GDALClose(ds);
}

/* Window size from the "bounds" array of metadata.json. */
static void parse_bounds(const char *json, const char *path, int *w, int *h)
{
    const char *p = strstr(json, "\"bounds\"");
    long b[4];

    if (!p || !(p = strchr(p, '[')) ||
        sscanf(p, "[%ld , %ld , %ld , %ld", &b[0], &b[1], &b[2], &b[3]) != 4)
        fatal("No bounds in <%s>", path);
    *w = (int)(b[2] - b[0]);
    *h = (int)(b[3] - b[1]);
    if (*w <= 0 || *h <= 0)
        fatal("Invalid bounds in <%s>", path);
}

static void load_window(struct olmo_window *win, const char *wdir,
                        const char *s2_layer, const char *label_layer,
                        enum olmo_layer_order order)
{
    char path[4096], *json, *split;
    size_t plane;

    xsnprintf(path, sizeof(path), "%s/metadata.json", wdir);
    json = read_file(path);
    split = json_string(json, "split");
    win->split = parse_split(split);
    parse_bounds(json, path, &win->width, &win->height);
    free(split);
    free(json);

    win->ntime = list_layers(win, wdir, s2_layer, order);
    if (win->ntime == 0)
        fatal("Window <%s> has no completed <%s> layer", win->name, s2_layer);

    plane = (size_t)win->width * win->height;
    win->s2 = xmalloc(sizeof(uint16_t) * win->ntime * OLMO_NBANDS * plane);
    for (int t = 0; t < win->ntime; t++) {
        xsnprintf(path, sizeof(path),
                  "%s/layers/%s/" S2_BANDS_DIR "/geotiff.tif", wdir,
                  win->layer[t]);
        read_raster(path, win->width, win->height, OLMO_NBANDS, GDT_UInt16,
                    win->s2 + (size_t)t * OLMO_NBANDS * plane);
    }

    win->label = xmalloc(plane);
    xsnprintf(path, sizeof(path), "%s/layers/%s/%s/geotiff.tif", wdir,
              label_layer, label_layer);
    read_raster(path, win->width, win->height, 1, GDT_Byte, win->label);
    for (size_t i = 0; i < plane; i++)
        if (win->label[i] > 1 && win->label[i] != OLMO_LABEL_NODATA)
            fatal("Label <%s> has class value %d; only 0, 1 and %d (nodata) "
                  "are expected",
                  path, win->label[i], OLMO_LABEL_NODATA);
}

static int cmp_window(const void *a, const void *b)
{
    return strcmp(((const struct olmo_window *)a)->name,
                  ((const struct olmo_window *)b)->name);
}

void olmo_dataset_load(struct olmo_dataset *ds, const char *dir,
                       const char *group, const char *s2_layer,
                       const char *label_layer, enum olmo_layer_order order)
{
    char gdir[4096];
    struct dirent *de;
    DIR *d;
    int cap = 0;

    memset(ds, 0, sizeof(*ds));
    GDALAllRegister();
    xsnprintf(gdir, sizeof(gdir), "%s/windows/%s", dir, group);
    d = opendir(gdir);
    if (!d)
        fatal("Cannot open window group <%s>", gdir);
    while ((de = readdir(d))) {
        char meta[4096];

        if (de->d_name[0] == '.')
            continue;
        xsnprintf(meta, sizeof(meta), "%s/%s/metadata.json", gdir, de->d_name);
        if (!exists(meta))
            continue;
        if (ds->nwindows == cap) {
            cap = cap ? 2 * cap : 64;
            ds->windows = xrealloc(ds->windows, sizeof(*ds->windows) * cap);
        }
        memset(&ds->windows[ds->nwindows], 0, sizeof(*ds->windows));
        ds->windows[ds->nwindows++].name = xstrdup(de->d_name);
    }
    closedir(d);
    if (ds->nwindows == 0)
        fatal("No windows in <%s>", gdir);
    /* Sorted by name so that runs are reproducible whatever the
     * directory order; the training loop shuffles anyway. */
    qsort(ds->windows, ds->nwindows, sizeof(*ds->windows), cmp_window);

    for (int i = 0; i < ds->nwindows; i++) {
        struct olmo_window *win = &ds->windows[i];
        char wdir[4096];

        xsnprintf(wdir, sizeof(wdir), "%s/%s", gdir, win->name);
        load_window(win, wdir, s2_layer, label_layer, order);
        if (i == 0)
            ds->ntime = win->ntime;
        else if (win->ntime != ds->ntime)
            fatal("Window <%s> has %d timesteps, <%s> has %d: mixed lengths "
                  "need the masked encoder path, which is not implemented",
                  win->name, win->ntime, ds->windows[0].name, ds->ntime);
    }
}

void olmo_norm_load(struct olmo_norm *n, const char *path)
{
    static const char *bands[OLMO_NBANDS] = {"B02", "B03", "B04", "B08",
                                             "B05", "B06", "B07", "B8A",
                                             "B11", "B12", "B01", "B09"};
    FILE *fp = fopen(path, "r");
    char line[256], band[16];
    double mean, std;
    int b = 0;

    if (!fp)
        fatal("Cannot open normalisation table <%s>", path);
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n')
            continue;
        if (sscanf(line, "%15s %lf %lf", band, &mean, &std) != 3)
            fatal("Invalid line in <%s>: %s", path, line);
        if (b == OLMO_NBANDS || strcmp(band, bands[b]) != 0)
            fatal("<%s>: band <%s> found where <%s> was expected", path, band,
                  b < OLMO_NBANDS ? bands[b] : "end of file");
        if (!(std > 0.0))
            fatal("<%s>: band <%s> has std %g", path, band, std);
        n->lo[b] = (float)(mean - 2.0 * std);
        n->inv[b] = (float)(1.0 / (4.0 * std));
        b++;
    }
    fclose(fp);
    if (b != OLMO_NBANDS)
        fatal("<%s> has %d bands, expected %d", path, b, OLMO_NBANDS);
}

void olmo_make_crop(const struct olmo_window *win, const struct olmo_norm *n,
                    int ox, int oy, int size, int flip_h, int flip_v,
                    float *out, uint8_t *label)
{
    const size_t plane = (size_t)win->width * win->height;

    if (ox < 0 || oy < 0 || ox + size > win->width || oy + size > win->height)
        fatal("Crop %dx%d at (%d,%d) is outside window <%s> (%dx%d)", size,
              size, ox, oy, win->name, win->width, win->height);
    for (int t = 0; t < win->ntime; t++)
        for (int b = 0; b < OLMO_NBANDS; b++) {
            const uint16_t *src =
                win->s2 + ((size_t)t * OLMO_NBANDS + b) * plane;
            float *dst = out + ((size_t)t * OLMO_NBANDS + b) * size * size;

            for (int r = 0; r < size; r++) {
                const int sr = oy + (flip_v ? size - 1 - r : r);

                for (int c = 0; c < size; c++) {
                    const int sc = ox + (flip_h ? size - 1 - c : c);

                    dst[r * size + c] =
                        ((float)src[(size_t)sr * win->width + sc] - n->lo[b]) *
                        n->inv[b];
                }
            }
        }
    if (!label)
        return;
    for (int r = 0; r < size; r++) {
        const int sr = oy + (flip_v ? size - 1 - r : r);

        for (int c = 0; c < size; c++) {
            const int sc = ox + (flip_h ? size - 1 - c : c);

            label[r * size + c] = win->label[(size_t)sr * win->width + sc];
        }
    }
}

const struct olmo_window *olmo_dataset_find(const struct olmo_dataset *ds,
                                            const char *name)
{
    for (int i = 0; i < ds->nwindows; i++)
        if (strcmp(ds->windows[i].name, name) == 0)
            return &ds->windows[i];
    return NULL;
}

void olmo_dataset_free(struct olmo_dataset *ds)
{
    for (int i = 0; i < ds->nwindows; i++) {
        struct olmo_window *win = &ds->windows[i];

        free(win->name);
        for (int t = 0; t < win->ntime; t++)
            free(win->layer[t]);
        free(win->s2);
        free(win->label);
    }
    free(ds->windows);
    memset(ds, 0, sizeof(*ds));
}
