#include <cuda_runtime.h>
#include <cstdint>

#include "pi_type.hpp"

__host__ __device__ __forceinline__ bool is_aligned_16(const void *p)
{
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

__device__ __forceinline__ float4 load_float4(const float *p)
{
    return *reinterpret_cast<const float4 *>(p);
}

__device__ __forceinline__ void store_float4(float *p, float4 v)
{
    *reinterpret_cast<float4 *>(p) = v;
}

__device__ __forceinline__ float4 make_float4_0()
{
    float4 v;
    v.x = v.y = v.z = v.w = 0.0f;
    return v;
}

__device__ __forceinline__ float4 load_float4_safe(const float *p, int n_remain)
{
    if (n_remain >= 4)
        return load_float4(p);
    float4 v = make_float4_0();
    if (n_remain > 0)
        v.x = p[0];
    if (n_remain > 1)
        v.y = p[1];
    if (n_remain > 2)
        v.z = p[2];
    if (n_remain > 3)
        v.w = p[3];
    return v;
}

__device__ __forceinline__ void store_float4_safe(float *p, float4 v, int n_remain)
{
    if (n_remain >= 4)
    {
        store_float4(p, v);
        return;
    }
    if (n_remain > 0)
        p[0] = v.x;
    if (n_remain > 1)
        p[1] = v.y;
    if (n_remain > 2)
        p[2] = v.z;
    if (n_remain > 3)
        p[3] = v.w;
}

template <typename T, int BM, int BK, int BN>
__global__ void gemm_kernel_generic(int M, int K, int N,
                                    const T *__restrict__ A,
                                    const T *__restrict__ B,
                                    T *__restrict__ C,
                                    T alpha, T beta)
{
    __shared__ T A_tile[BM][BK];
    __shared__ T B_tile[BK][BN];

    int c_tile_row = blockIdx.y;
    int c_tile_col = blockIdx.x;
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int row = c_tile_row * BM + local_row;
    int col = c_tile_col * BN + local_col;

    T acc = 0;
    for (int k0 = 0; k0 < K; k0 += BK)
    {
        int a_row = row;
        int a_col = k0 + local_col;
        int b_row = k0 + local_row;
        int b_col = col;

        A_tile[local_row][local_col] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : T(0);
        B_tile[local_row][local_col] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : T(0);

        __syncthreads();

        for (int t = 0; t < BK; ++t)
            acc += A_tile[local_row][t] * B_tile[t][local_col];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc * alpha + beta * C[row * N + col];
}

template <int BM, int BK, int BN, int TM, int TN>
__global__ void gemm_kernel_fp32_fast(
    int M, int K, int N,
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    float alpha, float beta)
{
    constexpr int BK_PAD = BK + 4;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int row_base = block_row + local_row * TM;
    int col_base = block_col + local_col * TN;

    __shared__ __align__(16) float A_tile[BM][BK_PAD];
    __shared__ __align__(16) float4 B_tile[BK][BN / 4];

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.f;

    int tid = local_row * blockDim.x + local_col;
    int nthreads = blockDim.x * blockDim.y;

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        {
            int idx = tid;
            int r = idx / (BK / 4);
            int c4 = (idx % (BK / 4)) * 4;

            int a_row = block_row + r;
            int a_col = k0 + c4;

            float4 v4 = load_float4(&A[a_row * K + a_col]);
            *reinterpret_cast<float4 *>(&A_tile[r][c4]) = v4;
        }

        for (int idx = tid; idx < BK * (BN / 4); idx += nthreads)
        {
            int r = idx / (BN / 4);
            int c4 = (idx % (BN / 4)) * 4;

            int b_row = k0 + r;
            int b_col = block_col + c4;

            B_tile[r][c4 / 4] = load_float4(&B[b_row * N + b_col]);
        }

        __syncthreads();

#pragma unroll
        for (int t = 0; t < BK; ++t)
        {
            float a[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = A_tile[local_row * TM + i][t];

            float4 bv = B_tile[t][local_col];
            float b[TN] = {bv.x, bv.y, bv.z, bv.w};

#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a[i] * b[j];
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        float *p = C + (row_base + i) * N + col_base;

        float4 outv;
        outv.x = acc[i][0] * alpha;
        outv.y = acc[i][1] * alpha;
        outv.z = acc[i][2] * alpha;
        outv.w = acc[i][3] * alpha;

        if (beta != 0.0f)
        {
            float4 cv = load_float4(p);
            outv.x += beta * cv.x;
            outv.y += beta * cv.y;
            outv.z += beta * cv.z;
            outv.w += beta * cv.w;
        }
        store_float4(p, outv);
    }
}

template <int BM, int BK, int BN, int TM, int TN>
__global__ void gemm_kernel_fp32_safe(
    int M, int K, int N,
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    float alpha, float beta)
{
    constexpr int BK_PAD = BK + 4;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int row_base = block_row + local_row * TM;
    int col_base = block_col + local_col * TN;

    __shared__ __align__(16) float A_tile[BM][BK_PAD];
    __shared__ __align__(16) float4 B_tile[BK][BN / 4];

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.f;

    int tid = local_row * blockDim.x + local_col;
    int nthreads = blockDim.x * blockDim.y;

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        {
            int idx = tid;
            int r = idx / (BK / 4);
            int c4 = (idx % (BK / 4)) * 4;

            int a_row = block_row + r;
            int a_col = k0 + c4;

            float4 v4 = make_float4_0();
            if (a_row < M)
                v4 = load_float4_safe(&A[a_row * K + a_col], K - a_col);

            *reinterpret_cast<float4 *>(&A_tile[r][c4]) = v4;
        }

        for (int idx = tid; idx < BK * (BN / 4); idx += nthreads)
        {
            int r = idx / (BN / 4);
            int c4 = (idx % (BN / 4)) * 4;

            int b_row = k0 + r;
            int b_col = block_col + c4;

            float4 v4 = make_float4_0();
            if (b_row < K)
                v4 = load_float4_safe(&B[b_row * N + b_col], N - b_col);

            B_tile[r][c4 / 4] = v4;
        }

        __syncthreads();

        int t_end = min(BK, K - k0);
#pragma unroll
        for (int t = 0; t < t_end; ++t)
        {
            float a[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = A_tile[local_row * TM + i][t];

            float4 bv = B_tile[t][local_col];
            float b[TN] = {bv.x, bv.y, bv.z, bv.w};

#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a[i] * b[j];
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        int row = row_base + i;
        if (row >= M)
            continue;

        float *p = C + row * N + col_base;
        int n_remain = N - col_base;
        if (n_remain <= 0)
            continue;

        float4 outv;
        outv.x = acc[i][0] * alpha;
        outv.y = acc[i][1] * alpha;
        outv.z = acc[i][2] * alpha;
        outv.w = acc[i][3] * alpha;

        if (beta != 0.0f)
        {
            float4 cv = load_float4_safe(p, n_remain);
            outv.x += beta * cv.x;
            outv.y += beta * cv.y;
            outv.z += beta * cv.z;
            outv.w += beta * cv.w;
        }
        store_float4_safe(p, outv, n_remain);
    }
}

piState pi_cuda_gemm_fp32(
    float *__restrict__ A,
    float *__restrict__ B,
    float *__restrict__ C,
    float alpha,
    float beta,
    int M, int K, int N)
{
    constexpr int BM = 32;
    constexpr int BK = 32;
    constexpr int BN = 128;
    constexpr int TM = 4;
    constexpr int TN = 4;

    dim3 block(BN / TN, BM / TM);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM);

    bool full_tile =
        (M % BM == 0) &&
        (N % BN == 0) &&
        (K % BK == 0);

    bool vec_ok =
        (TN == 4) &&
        (K % 4 == 0) &&
        (N % 4 == 0) &&
        is_aligned_16(A) &&
        is_aligned_16(B) &&
        is_aligned_16(C);

    if (full_tile && vec_ok)
    {
        gemm_kernel_fp32_fast<
            BM, BK, BN, TM, TN><<<grid, block>>>(
            M, K, N, A, B, C, alpha, beta);
    }
    else
    {
        gemm_kernel_fp32_safe<
            BM, BK, BN, TM, TN><<<grid, block>>>(
            M, K, N, A, B, C, alpha, beta);
    }

    return piSuccess;
}

piState pi_cuda_gemm_fp64(
    double *__restrict__ A,
    double *__restrict__ B,
    double *__restrict__ C,
    double alpha,
    double beta,
    int M, int K, int N)
{
    constexpr int BM = 16;
    constexpr int BK = 16;
    constexpr int BN = 16;

    dim3 block(BN, BM);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM);

    gemm_kernel_generic<double, BM, BK, BN><<<grid, block>>>(
        M, K, N, A, B, C, alpha, beta);

    return piSuccess;
}
