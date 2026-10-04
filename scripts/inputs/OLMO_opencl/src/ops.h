/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Host-side launchers of the OpenCL kernels (GEMM, LayerNorm,
 *               softmax), after the helpers of i.sam.opencl.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#ifndef OLMO_CL_OPS_H
#define OLMO_CL_OPS_H

#include "ocl_backend.h"

#define ACT_NONE 0
#define ACT_GELU 1
#define ACT_RELU 2

struct olmo_ops {
    struct ocl_backend *ocl;
    cl_kernel k_gemm, k_gemm_big, k_ln, k_softmax, k_add;
};

/* One GEMM operand: buffer, element offset, leading dimension and the
 * two batch strides. */
struct opnd {
    cl_mem buf;
    cl_long off;
    cl_int ld;
    cl_long s0, s1;
};

struct opnd mat(cl_mem buf, long off, int ld);
struct opnd bmat(cl_mem buf, long off, int ld, long s0, long s1);

void ops_init(struct olmo_ops *o, struct ocl_backend *ocl);
void ops_free(struct olmo_ops *o);

/* Print the accumulated kernel times (OLMO_CL_PROFILE set). */
void ops_profile_report(void);

/* C = act(alpha A op(B) + bias) + R over nb batches; batch b uses
 * (b / nb1, b % nb1) with the operands' strides s0 and s1. transb != 0
 * reads B as [N, K] (PyTorch Linear weight layout). */
void gemm(struct olmo_ops *o, int M, int N, int K, struct opnd A, struct opnd B,
          int transb, struct opnd C, int nb, int nb1, cl_mem bias, cl_mem R,
          long roff, float alpha, int act);

/* Y[rows, out] = act(X[rows, in] W^T + b) (+ Y when residual). */
void linear(struct olmo_ops *o, int rows, int in, int out, cl_mem X, long xoff,
            cl_mem W, cl_mem b, cl_mem Y, long yoff, int act, int residual);

/* Row LayerNorm over C channels; Y may alias X. */
void layernorm(struct olmo_ops *o, long rows, int C, cl_mem X, cl_mem Y,
               cl_mem g, cl_mem b, float eps, int act);

/* X[i] += E[i] for i < n. */
void add_inplace(struct olmo_ops *o, cl_mem X, cl_mem E, int n);

/* Wait for the queued kernels. */
void ops_finish(struct olmo_ops *o);

/* In-place softmax of rows rows of L scores. */
void softmax_rows(struct olmo_ops *o, cl_mem S, long rows, int L);

#endif /* OLMO_CL_OPS_H */
