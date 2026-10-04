/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Kernel self-test against plain C loops.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#ifndef OLMO_CL_SELFTEST_H
#define OLMO_CL_SELFTEST_H

#include "ops.h"

/* Run every kernel on random data shaped like the OlmoEarth-Tiny encoder
 * (tokens attention rows) and compare with double-precision C loops.
 * Returns the number of failed checks. */
int olmo_selftest(struct olmo_ops *o, int tokens);

#endif /* OLMO_CL_SELFTEST_H */
