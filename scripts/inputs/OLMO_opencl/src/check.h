/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Stage-by-stage comparison with PyTorch golden dumps
 *               (tools/make_golden.py).
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#ifndef OLMO_CL_CHECK_H
#define OLMO_CL_CHECK_H

#include "dataset.h"
#include "encoder.h"

/* Run the encoder on the golden sample in dir and compare every stage.
 * With ds and norm (both or neither), the input crop is also rebuilt
 * from the dataset and compared, then used. Returns the number of
 * failed stages. */
int olmo_check(struct olmo_encoder *enc, const char *dir,
               const struct olmo_dataset *ds, const struct olmo_norm *norm);

/* Compare the head (probabilities, loss, dW, db) with the golden sample
 * in dir, on features [D][g][g], or on the golden features.f32 if NULL.
 * Returns the number of failed checks. */
int olmo_check_head(const char *dir, const float *features, int D, int g);

#endif /* OLMO_CL_CHECK_H */
