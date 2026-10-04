/****************************************************************************
 *
 * MODULE:       olmo_cl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      OpenCL kernels for the OlmoEarth encoder. GEMM, LayerNorm
 *               and the reductions come from i.sam.opencl. Embedded into
 *               the binary at build time (see Makefile).
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

/* Tensor layout convention: activations are token-major (row = spatial
 * position or prompt token, column = channel), which makes every
 * nn.Linear and 1x1 convolution a plain row-major GEMM against the
 * PyTorch [out, in] weight matrix.
 *
 * Element-wise kernels index with 32-bit integers: 64-bit division is
 * emulated on GPUs and was measured to dominate their run time. The
 * host keeps every tensor below 2^31 elements. */

#define ACT_NONE 0
#define ACT_GELU 1
#define ACT_RELU 2

float gelu_erf(float x)
{
    return 0.5f * x * (1.0f + erf(x * 0.70710678118654752f));
}

float apply_act(float v, int act)
{
    if (act == ACT_GELU)
        return gelu_erf(v);
    if (act == ACT_RELU)
        return fmax(v, 0.0f);
    return v;
}

/* Batched GEMM: C = act(alpha * A * op(B) + bias) + R.
 *
 * A is M x K row-major (lda). With transb != 0, B is stored N x K (the
 * PyTorch Linear weight layout, B(k, n) = B[n * ldb + k]); otherwise B
 * is K x N (B(k, n) = B[k * ldb + n]). Batch index b = get_group_id(2)
 * is split into (b / nb1, b % nb1), each part with its own element
 * stride per operand; a stride of 0 shares an operand across that batch
 * axis. Element offsets aoff/boff/coff/roff select sub-matrices of the
 * buffers. R, when used, has C's layout and may alias C for in-place
 * residual updates.
 *
 * Generic version: 64 x 64 output tile per work-group of 16 x 16 items,
 * each item computing a 4 x 4 register block. */
__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void
gemm(const int M, const int N, const int K, __global const float *A,
     const long aoff, const int lda, const long sa0, const long sa1,
     __global const float *B, const long boff, const int ldb, const long sb0,
     const long sb1, const int transb, __global float *C, const long coff,
     const int ldc, const long sc0, const long sc1, const int nb1,
     __global const float *bias, __global const float *R, const long roff,
     const float alpha, const int act)
{
    __local float As[16][65];
    __local float Bs[16][65];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int tid = ty * 16 + tx;
    const int n0 = get_group_id(0) * 64, m0 = get_group_id(1) * 64;
    const int b = get_group_id(2);
    const long b0 = b / nb1, b1 = b % nb1;
    float acc[4][4];
    int i, j, k, l;

    A += aoff + b0 * sa0 + b1 * sa1;
    B += boff + b0 * sb0 + b1 * sb1;
    C += coff + b0 * sc0 + b1 * sc1;
    if (R)
        R += roff + b0 * sc0 + b1 * sc1;

    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (l = 0; l < 4; l++) {
            const int idx = tid + l * 256;
            const int r = idx >> 4, c = idx & 15;
            const int gm = m0 + r, gk = k0 + c;

            As[c][r] = (gm < M && gk < K) ? A[(long)gm * lda + gk] : 0.0f;
            if (transb) {
                const int gn = n0 + r;

                Bs[c][r] = (gn < N && gk < K) ? B[(long)gn * ldb + gk] : 0.0f;
            }
            else {
                const int kk = idx >> 6, nn = idx & 63;
                const int gk2 = k0 + kk, gn = n0 + nn;

                Bs[kk][nn] =
                    (gn < N && gk2 < K) ? B[(long)gk2 * ldb + gn] : 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (k = 0; k < 16; k++) {
            float a[4], bb[4];

            for (i = 0; i < 4; i++)
                a[i] = As[k][ty + 16 * i];
            for (j = 0; j < 4; j++)
                bb[j] = Bs[k][tx + 16 * j];
            for (i = 0; i < 4; i++)
                for (j = 0; j < 4; j++)
                    acc[i][j] = mad(a[i], bb[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (i = 0; i < 4; i++) {
        const int m = m0 + ty + 16 * i;

        if (m >= M)
            continue;
        for (j = 0; j < 4; j++) {
            const int n = n0 + tx + 16 * j;
            float v;

            if (n >= N)
                continue;
            v = alpha * acc[i][j];
            if (bias)
                v += bias[n];
            v = apply_act(v, act);
            if (R)
                v += R[(long)m * ldc + n];
            C[(long)m * ldc + n] = v;
        }
    }
}

/* Epilogue of gemm_big: one row of an 8 x 8 register block. */
void store_row8(__global float *C, __global const float *R,
                __global const float *bias, const int m, const int n,
                const int M, const int N, const int ldc, const float alpha,
                const int act, const float4 lo, const float4 hi)
{
    const float v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};

    if (m >= M)
        return;
    for (int j = 0; j < 8; j++) {
        float x;

        if (n + j >= N)
            return;
        x = alpha * v[j];
        if (bias)
            x += bias[n + j];
        x = apply_act(x, act);
        if (R)
            x += R[(long)m * ldc + n + j];
        C[(long)m * ldc + n + j] = x;
    }
}

/* Rank-1 update of the 8 x 8 block row r from A column value s. */
#define ROW_FMA(r, s)                                                          \
    c##r##0 = mad((float4)(s), b0, c##r##0);                                   \
    c##r##1 = mad((float4)(s), b1, c##r##1)

/* Same contract as gemm, for large, 16-byte aligned operands: 128 x 128
 * output tile per 16 x 16 work-group, 8 x 8 register block per item
 * held in explicit float4 accumulators (private arrays of that size end
 * up in scratch memory with Mesa, costing a factor of 70), float4 loads.
 * Requires K, lda, ldb and all A/B offsets and batch strides to be
 * multiples of 4 (checked by the host). */
__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void
gemm_big(const int M, const int N, const int K, __global const float *A,
         const long aoff, const int lda, const long sa0, const long sa1,
         __global const float *B, const long boff, const int ldb,
         const long sb0, const long sb1, const int transb, __global float *C,
         const long coff, const int ldc, const long sc0, const long sc1,
         const int nb1, __global const float *bias, __global const float *R,
         const long roff, const float alpha, const int act)
{
    __local float As[16][132];
    __local float Bs[16][132];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int tid = ty * 16 + tx;
    const int n0 = get_group_id(0) * 128, m0 = get_group_id(1) * 128;
    const int b = get_group_id(2);
    const long bb0 = b / nb1, bb1 = b % nb1;
    float4 c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
    float4 c20 = 0.0f, c21 = 0.0f, c30 = 0.0f, c31 = 0.0f;
    float4 c40 = 0.0f, c41 = 0.0f, c50 = 0.0f, c51 = 0.0f;
    float4 c60 = 0.0f, c61 = 0.0f, c70 = 0.0f, c71 = 0.0f;

    A += aoff + bb0 * sa0 + bb1 * sa1;
    B += boff + bb0 * sb0 + bb1 * sb1;
    C += coff + bb0 * sc0 + bb1 * sc1;
    if (R)
        R += roff + bb0 * sc0 + bb1 * sc1;

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int l = 0; l < 2; l++) {
            const int idx = tid + l * 256;
            const int r = idx >> 2, kq = (idx & 3) * 4;
            const int gm = m0 + r, gk = k0 + kq;
            float4 v = 0.0f;

            if (gm < M && gk < K)
                v = vload4(0, A + (long)gm * lda + gk);
            As[kq][r] = v.x;
            As[kq + 1][r] = v.y;
            As[kq + 2][r] = v.z;
            As[kq + 3][r] = v.w;

            if (transb) {
                const int gn = n0 + r;

                v = 0.0f;
                if (gn < N && gk < K)
                    v = vload4(0, B + (long)gn * ldb + gk);
                Bs[kq][r] = v.x;
                Bs[kq + 1][r] = v.y;
                Bs[kq + 2][r] = v.z;
                Bs[kq + 3][r] = v.w;
            }
            else {
                const int kk = idx >> 5, nq = (idx & 31) * 4;
                const int gk2 = k0 + kk, gn = n0 + nq;

                v = 0.0f;
                if (gk2 < K) {
                    __global const float *bp = B + (long)gk2 * ldb + gn;

                    if (gn + 3 < N)
                        v = vload4(0, bp);
                    else {
                        if (gn < N)
                            v.x = bp[0];
                        if (gn + 1 < N)
                            v.y = bp[1];
                        if (gn + 2 < N)
                            v.z = bp[2];
                    }
                }
                vstore4(v, 0, &Bs[kk][nq]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < 16; k++) {
            const float4 a0 = vload4(0, &As[k][ty * 8]);
            const float4 a1 = vload4(0, &As[k][ty * 8 + 4]);
            const float4 b0 = vload4(0, &Bs[k][tx * 8]);
            const float4 b1 = vload4(0, &Bs[k][tx * 8 + 4]);

            ROW_FMA(0, a0.x);
            ROW_FMA(1, a0.y);
            ROW_FMA(2, a0.z);
            ROW_FMA(3, a0.w);
            ROW_FMA(4, a1.x);
            ROW_FMA(5, a1.y);
            ROW_FMA(6, a1.z);
            ROW_FMA(7, a1.w);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    store_row8(C, R, bias, m0 + ty * 8 + 0, n0 + tx * 8, M, N, ldc, alpha, act,
               c00, c01);
    store_row8(C, R, bias, m0 + ty * 8 + 1, n0 + tx * 8, M, N, ldc, alpha, act,
               c10, c11);
    store_row8(C, R, bias, m0 + ty * 8 + 2, n0 + tx * 8, M, N, ldc, alpha, act,
               c20, c21);
    store_row8(C, R, bias, m0 + ty * 8 + 3, n0 + tx * 8, M, N, ldc, alpha, act,
               c30, c31);
    store_row8(C, R, bias, m0 + ty * 8 + 4, n0 + tx * 8, M, N, ldc, alpha, act,
               c40, c41);
    store_row8(C, R, bias, m0 + ty * 8 + 5, n0 + tx * 8, M, N, ldc, alpha, act,
               c50, c51);
    store_row8(C, R, bias, m0 + ty * 8 + 6, n0 + tx * 8, M, N, ldc, alpha, act,
               c60, c61);
    store_row8(C, R, bias, m0 + ty * 8 + 7, n0 + tx * 8, M, N, ldc, alpha, act,
               c70, c71);
}

/* Sum and max of val over a work-group of n items (power of two). */
float wg_sum(float val, __local float *scratch, const int n)
{
    const int lid = get_local_id(0);

    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = n / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return val;
}

float wg_max(float val, __local float *scratch, const int n)
{
    const int lid = get_local_id(0);

    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = n / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return val;
}

/* Row-wise LayerNorm over C channels, Y may alias X. The work-group
 * size (256 for the encoder's wide rows, 64 for the decoder's 64- and
 * 256-channel rows) is the local size chosen by the host. */
__kernel void layernorm(__global const float *X, __global float *Y,
                        __global const float *g, __global const float *bt,
                        const int C, const float eps, const int act)
{
    __local float scratch[256];
    const int lid = get_local_id(0), n = get_local_size(0);
    const long row = get_group_id(0);
    __global const float *x = X + row * C;
    __global float *y = Y + row * C;
    float s = 0.0f, mean, var;
    int c;

    for (c = lid; c < C; c += n)
        s += x[c];
    mean = wg_sum(s, scratch, n) / C;
    s = 0.0f;
    for (c = lid; c < C; c += n) {
        const float d = x[c] - mean;

        s += d * d;
    }
    var = wg_sum(s, scratch, n) / C;
    s = rsqrt(var + eps);
    for (c = lid; c < C; c += n)
        y[c] = apply_act((x[c] - mean) * s * g[c] + bt[c], act);
}

/* In-place row softmax of attention scores, one 256-item work-group per
 * row of L scores. The scale is already applied by the QK^T GEMM. */
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void
softmax_rows(__global float *S, const int L)
{
    __local float scratch[256];
    const int lid = get_local_id(0);
    __global float *s = S + (long)get_group_id(0) * L;
    float m = -INFINITY, sum = 0.0f, v;
    int j;

    for (j = lid; j < L; j += 256)
        m = fmax(m, s[j]);
    m = wg_max(m, scratch, 256);
    for (j = lid; j < L; j += 256) {
        v = exp(s[j] - m);
        s[j] = v;
        sum += v;
    }
    sum = wg_sum(sum, scratch, 256);
    v = 1.0f / sum;
    for (j = lid; j < L; j += 256)
        s[j] *= v;
}

/* X[i] += E[i] for i < n: adds the precomputed encoding table to the
 * patch-embedded tokens. */
__kernel void add_inplace(__global float *X, __global const float *E,
                          const int n)
{
    const int i = get_global_id(0);

    if (i < n)
        X[i] += E[i];
}
