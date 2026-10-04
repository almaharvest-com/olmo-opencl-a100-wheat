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

#ifndef OLMO_CL_DATASET_H
#define OLMO_CL_DATASET_H

#include <stdint.h>

#include "weights.h"

#define OLMO_MAX_TIMESTEPS 12
#define OLMO_LABEL_NODATA  255

enum olmo_split { SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_OTHER };

struct olmo_window {
    char *name;
    enum olmo_split split;
    int width, height;
    int ntime;
    /* Layer directory of each timestep, in rslearn's stacking order. */
    char *layer[OLMO_MAX_TIMESTEPS];
    uint16_t *s2;   /* [ntime][OLMO_NBANDS][height][width] */
    uint8_t *label; /* [height][width], 0, 1 or OLMO_LABEL_NODATA */
};

struct olmo_dataset {
    int nwindows;
    struct olmo_window *windows;
    int ntime; /* Common number of timesteps. */
};

enum olmo_layer_order {
    /* Directory order, as rslearn's list_completed_layers() uses. */
    ORDER_READDIR,
    /* sentinel2, sentinel2.1, ... by item group index. */
    ORDER_INDEX
};

/* Load every window of dir/windows/group. s2_layer and label_layer are
 * the rslearn layer names ("sentinel2", "label"). Fatal on missing or
 * inconsistent data: every window must have the same number of
 * timesteps, 12 bands and matching raster sizes. */
void olmo_dataset_load(struct olmo_dataset *ds, const char *dir,
                       const char *group, const char *s2_layer,
                       const char *label_layer, enum olmo_layer_order order);

void olmo_dataset_free(struct olmo_dataset *ds);

const char *olmo_split_name(enum olmo_split s);

/* rslearn OlmoEarthNormalize: x' = (x - lo) * inv, lo = mean - 2 std,
 * inv = 1 / (4 std), per band in OlmoEarth band order. */
struct olmo_norm {
    float lo[OLMO_NBANDS], inv[OLMO_NBANDS];
};

/* Read data/s2_norm.tsv (band, mean, std), checking the band order. */
void olmo_norm_load(struct olmo_norm *n, const char *path);

/* Normalised size x size crop at (ox, oy) of a window, optionally
 * flipped left-right (flip_h) and/or top-bottom (flip_v) as rslearn's
 * Flip transform does. out is [ntime][OLMO_NBANDS][size][size]. The
 * label crop (same flips) goes to label if not NULL, [size][size]. */
void olmo_make_crop(const struct olmo_window *win, const struct olmo_norm *n,
                    int ox, int oy, int size, int flip_h, int flip_v,
                    float *out, uint8_t *label);

/* Return the window called name, or NULL. */
const struct olmo_window *olmo_dataset_find(const struct olmo_dataset *ds,
                                            const char *name);

#endif /* OLMO_CL_DATASET_H */
