/* SPDX-License-Identifier: Unlicense */

/* AdamW and ReduceLROnPlateau against PyTorch: run by tests/test_optim.py,
 * which feeds the same deterministic gradients and metrics to torch. */

#include <math.h>
#include <stdio.h>

#include "optim.h"

int main(void)
{
    enum { N = 8, STEPS = 300, EPOCHS = 60 };
    float p[N], g[N];
    struct adamw a;
    struct plateau s;
    double lr = 1e-3;

    for (int i = 0; i < N; i++)
        p[i] = 0.1f * (i - 3);
    adamw_init(&a, N, lr);
    for (int t = 0; t < STEPS; t++) {
        for (int i = 0; i < N; i++)
            g[i] = (float)(sin(0.37 * t + i) * (1.0 + 0.1 * i) + p[i]);
        adamw_step(&a, p, g);
    }
    for (int i = 0; i < N; i++)
        printf("%.9g\n", p[i]);

    plateau_init(&s, 0.2, 2, 10, 0.0);
    lr = 1e-4;
    for (int e = 0; e < EPOCHS; e++) {
        /* Decreasing, then flat with noise, then decreasing again. */
        double m =
            e < 10 ? 1.0 - 0.05 * e
                   : 0.5 + 0.00001 * sin(e) - (e > 40 ? 0.01 * (e - 40) : 0);

        lr = plateau_step(&s, m, lr);
        printf("%.9g\n", lr);
    }
    adamw_free(&a);
    return 0;
}
