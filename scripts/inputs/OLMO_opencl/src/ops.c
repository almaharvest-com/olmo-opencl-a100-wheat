/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Host-side launchers of the OpenCL kernels, after the
 *               helpers of i.sam.opencl (sam_model.c).
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <stdio.h>
#include <string.h>

#include "ops.h"
#include "util.h"

#define SETARG(k, i, v) ocl_check(clSetKernelArg(k, i, sizeof(v), &(v)), #k)

/* Accumulated kernel times when profiling is enabled. */
static struct {
    const char *what;
    double ms;
    long calls;
} prof[64];
static int nprof;

static void run(struct olmo_ops *o, cl_kernel k, int dim, const size_t *g,
                const size_t *l, const char *what)
{
    cl_event ev;
    cl_ulong t0, t1;
    int i;

    ocl_check(clEnqueueNDRangeKernel(o->ocl->queue, k, dim, NULL, g, l, 0, NULL,
                                     o->ocl->profile ? &ev : NULL),
              what);
    if (!o->ocl->profile)
        return;
    clWaitForEvents(1, &ev);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0,
                            NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1,
                            NULL);
    clReleaseEvent(ev);
    for (i = 0; i < nprof && strcmp(prof[i].what, what) != 0; i++)
        ;
    if (i == nprof && nprof < 64)
        prof[nprof++].what = what;
    if (i < 64) {
        prof[i].ms += (t1 - t0) * 1e-6;
        prof[i].calls++;
    }
}

void ops_profile_report(void)
{
    for (int i = 0; i < nprof; i++)
        msg_info("%-12s %6ld calls %10.1f ms", prof[i].what, prof[i].calls,
                 prof[i].ms);
}

struct opnd mat(cl_mem buf, long off, int ld)
{
    struct opnd a = {buf, off, ld, 0, 0};

    return a;
}

struct opnd bmat(cl_mem buf, long off, int ld, long s0, long s1)
{
    struct opnd a = {buf, off, ld, s0, s1};

    return a;
}

void ops_init(struct olmo_ops *o, struct ocl_backend *ocl)
{
    o->ocl = ocl;
    o->k_gemm = ocl_kernel(ocl, "gemm");
    o->k_gemm_big = ocl_kernel(ocl, "gemm_big");
    o->k_ln = ocl_kernel(ocl, "layernorm");
    o->k_softmax = ocl_kernel(ocl, "softmax_rows");
    o->k_add = ocl_kernel(ocl, "add_inplace");
}

void ops_free(struct olmo_ops *o)
{
    clReleaseKernel(o->k_gemm);
    clReleaseKernel(o->k_gemm_big);
    clReleaseKernel(o->k_ln);
    clReleaseKernel(o->k_softmax);
    clReleaseKernel(o->k_add);
    memset(o, 0, sizeof(*o));
}

void gemm(struct olmo_ops *o, int M, int N, int K, struct opnd A, struct opnd B,
          int transb, struct opnd C, int nb, int nb1, cl_mem bias, cl_mem R,
          long roff, float alpha, int act)
{
    cl_int cM = M, cN = N, cK = K, ct = transb, cnb1 = nb1, cact = act;
    cl_long croff = roff;
    cl_float calpha = alpha;
    size_t g[3], l[3] = {16, 16, 1};
    /* The 128 x 128 float4 kernel needs every A/B row start 16-byte
     * aligned; small or misaligned products use the generic kernel. */
    int big = M >= 128 && N >= 64 && K % 4 == 0 && A.ld % 4 == 0 &&
              A.off % 4 == 0 && A.s0 % 4 == 0 && A.s1 % 4 == 0 &&
              B.ld % 4 == 0 && B.off % 4 == 0 && B.s0 % 4 == 0 && B.s1 % 4 == 0;
    int tile = big ? 128 : 64;
    cl_kernel k = big ? o->k_gemm_big : o->k_gemm;

    SETARG(k, 0, cM);
    SETARG(k, 1, cN);
    SETARG(k, 2, cK);
    SETARG(k, 3, A.buf);
    SETARG(k, 4, A.off);
    SETARG(k, 5, A.ld);
    SETARG(k, 6, A.s0);
    SETARG(k, 7, A.s1);
    SETARG(k, 8, B.buf);
    SETARG(k, 9, B.off);
    SETARG(k, 10, B.ld);
    SETARG(k, 11, B.s0);
    SETARG(k, 12, B.s1);
    SETARG(k, 13, ct);
    SETARG(k, 14, C.buf);
    SETARG(k, 15, C.off);
    SETARG(k, 16, C.ld);
    SETARG(k, 17, C.s0);
    SETARG(k, 18, C.s1);
    SETARG(k, 19, cnb1);
    SETARG(k, 20, bias);
    SETARG(k, 21, R);
    SETARG(k, 22, croff);
    SETARG(k, 23, calpha);
    SETARG(k, 24, cact);
    g[0] = (size_t)((N + tile - 1) / tile) * 16;
    g[1] = (size_t)((M + tile - 1) / tile) * 16;
    g[2] = nb;
    run(o, k, 3, g, l, big ? "gemm_big" : "gemm");
}

void linear(struct olmo_ops *o, int rows, int in, int out, cl_mem X, long xoff,
            cl_mem W, cl_mem b, cl_mem Y, long yoff, int act, int residual)
{
    gemm(o, rows, out, in, mat(X, xoff, in), mat(W, 0, in), 1,
         mat(Y, yoff, out), 1, 1, b, residual ? Y : NULL, yoff, 1.0f, act);
}

void layernorm(struct olmo_ops *o, long rows, int C, cl_mem X, cl_mem Y,
               cl_mem g, cl_mem b, float eps, int act)
{
    cl_kernel k = o->k_ln;
    cl_int cC = C, cact = act;
    cl_float ceps = eps;
    /* 256 items per row for wide rows, one 64-item wavefront otherwise. */
    size_t ls = C >= 1024 ? 256 : 64, gs = rows * ls;

    SETARG(k, 0, X);
    SETARG(k, 1, Y);
    SETARG(k, 2, g);
    SETARG(k, 3, b);
    SETARG(k, 4, cC);
    SETARG(k, 5, ceps);
    SETARG(k, 6, cact);
    run(o, k, 1, &gs, &ls, "layernorm");
}

void softmax_rows(struct olmo_ops *o, cl_mem S, long rows, int L)
{
    cl_kernel k = o->k_softmax;
    cl_int cL = L;
    size_t ls = 256, gs = rows * ls;

    SETARG(k, 0, S);
    SETARG(k, 1, cL);
    run(o, k, 1, &gs, &ls, "softmax_rows");
}

void add_inplace(struct olmo_ops *o, cl_mem X, cl_mem E, int n)
{
    cl_kernel k = o->k_add;
    cl_int cn = n;
    size_t ls = 256, gs = ((size_t)n + ls - 1) / ls * ls;

    SETARG(k, 0, X);
    SETARG(k, 1, E);
    SETARG(k, 2, cn);
    run(o, k, 1, &gs, &ls, "add_inplace");
}

void ops_finish(struct olmo_ops *o)
{
    ocl_check(clFinish(o->ocl->queue), "clFinish");
}
