#include <cuda_runtime.h>
#include <cstdint>
#include "pi_type.h"

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

template <typename T>
__global__ void gemm_kernel(int M, int K, int N, T *__restrict__ A, T *__restrict__ B, T *C, T alpha, T beta)
{
    // 总体ijk循环
    //  每个线程处理C的一个元素，
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int row_index = index / N;
    int col_index = index % N;

    if (row_index < M && col_index < N)
    {
        // 整体缩放C：βC
        C[index] *= beta;
        // 矩阵乘：αAB
        T *C_row = C + row_index * N;
        T *A_row = A + row_index * K;
        T acc = 0;
#pragma unroll
        for (int k = 0; k < K; k++)
        {
            acc += B[k * N + col_index] * A_row[k];
        }
        // 累加
        C[index] += acc * alpha;
    }
}

template <typename T>
__global__ void gemm_kernel_v2(int M, int K, int N, T *__restrict__ A, T *__restrict__ B, T *C, T alpha, T beta)
{
    // 一个block对C的一个分块C_tile负责
    // 每个线程处理C_tile的一个元素
    // 总体ijk循环
    // 利用共享内存加速存储

    constexpr int BM = 16; // M方向的块长度
    constexpr int BK = 16; // K方向的块长度
    constexpr int BN = 16; // N方向的块长度

    // shared memory
    __shared__ T A_tile[BM][BK];
    __shared__ T B_tile[BK][BN];

    // 明确索引
    int c_tile_row = blockIdx.y; // Ctile在所有分块中的行索引
    int c_tile_col = blockIdx.x; // Ctile在所有分块中的列索引
    int local_row = threadIdx.y; // 线程负责Ctile元素的行索引
    int local_col = threadIdx.x; // 线程负责Ctile元素的列索引

    int row = c_tile_row * BM + local_row; // 线程负责元素在C中的行索引
    int col = c_tile_col * BN + local_col; // 线程负责元素在C中的列索引

    T acc = 0; // 寄存器用来作为累加器
    for (int k = 0; k < K; k += BK)
    {
        // 装载数据到共享内存
        int a_row = row;
        int a_col = k + local_col;

        int b_row = k + local_row;
        int b_col = col;
        if (a_row < M && a_col < K)
        {

            A_tile[local_row][local_col] = A[a_row * K + a_col];
        }
        else
        {
            A_tile[local_row][local_col] = (T)0;
        }

        if (b_row < K && b_col < N)
        {

            B_tile[local_row][local_col] = B[b_row * N + b_col];
        }
        else
        {

            B_tile[local_row][local_col] = (T)0;
        }
        __syncthreads();
        for (int t = 0; t < BK; ++t)
        {
            acc += A_tile[local_row][t] * B_tile[t][local_col];
        }

        __syncthreads();
    }
    if (row < M && col < N)
    {
        C[row * N + col] = acc * alpha + beta * C[row * N + col];
    }
}

template <typename T>
piState gemm_cuda_impl(int M, int K, int N, T *__restrict__ A, T *__restrict__ B, T *__restrict__ C, T alpha, T beta)
{
    // C=αAB+βC
    constexpr int BLOCK_SIZE = 256;
    int total = M * N;
    int GRID_SIZE = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gemm_kernel<T><<<GRID_SIZE, BLOCK_SIZE>>>(M, K, N, A, B, C, alpha, beta);
    return piSuccess;
}

piState piCudaGemmFp32(float *__restrict__ A,
                       float *__restrict__ B,
                       float *__restrict__ C,
                       float alpha,
                       float beta,
                       int M, int K, int N)
{
    return gemm_cuda_impl<float>(M, K, N, A, B, C, alpha, beta);
}

piState piCudaGemmFp64(double *__restrict__ A,
                       double *__restrict__ B,
                       double *__restrict__ C,
                       double alpha,
                       double beta,
                       int M, int K, int N)
{
    return gemm_cuda_impl<double>(M, K, N, A, B, C, alpha, beta);
}

template <typename T>
piState gemm_cuda_impl_v2(int M, int K, int N, T *__restrict__ A, T *__restrict__ B, T *__restrict__ C, T alpha, T beta)
{
    // C=αAB+βC
    constexpr int BM = 16;
    constexpr int BN = 16;

    dim3 block(BN, BM);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM);
    gemm_kernel_v2<T><<<grid, block>>>(M, K, N, A, B, C, alpha, beta);
    return piSuccess;
}

piState piCudaGemmFp32_v2(float *__restrict__ A,
                          float *__restrict__ B,
                          float *__restrict__ C,
                          float alpha,
                          float beta,
                          int M, int K, int N)
{
    return gemm_cuda_impl_v2<float>(M, K, N, A, B, C, alpha, beta);
}

piState piCudaGemmFp64_v2(double *__restrict__ A,
                          double *__restrict__ B,
                          double *__restrict__ C,
                          double alpha,
                          double beta,
                          int M, int K, int N)
{
    return gemm_cuda_impl_v2<double>(M, K, N, A, B, C, alpha, beta);
}

__global__ void gemm_kernel_v3(int M, int K, int N,
                               const float *__restrict__ A,
                               const float *__restrict__ B,
                               float *__restrict__ C,
                               float alpha, float beta)
{
    // 一个block对C的一个分块C_tile负责
    // 每个线程处理C_tile的一个子块(TM×TN个元素)，用寄存器累加
    // 总体ijk循环
    // 利用共享内存加速存储

    constexpr int BM = 32;                     // M方向的块长度
    constexpr int BK = 32;                     // K方向的块长度
    constexpr int BN = 32;                     // N方向的块长度
    constexpr int TM = 4;                      // 每线程在M方向计算的元素个数
    constexpr int TN = 4;                      // 每线程在N方向计算的元素个数
    int c_tile_row = blockIdx.y;               // Ctile在所有分块中的行索引
    int c_tile_col = blockIdx.x;               // Ctile在所有分块中的列索引
    int local_row = threadIdx.y;               // 线程负责Ctile子块的行索引(范围0..Mt/TM-1)
    int local_col = threadIdx.x;               // 线程负责Ctile子块的列索引(范围0..Nt/TN-1)
    int block_row = c_tile_row * BM;           // C分块左上角在C中的行索引
    int block_col = c_tile_col * BN;           // C分块左上角在C中的列索引
    int row_base = block_row + local_row * TM; // 线程子块左上角在C中的行索引
    int col_base = block_col + local_col * TN; // 线程子块左上角在C中的列索引

    // shared memory
    __shared__ float A_tile_t[BK][BM];
    __shared__ float B_tile[BK][BN];

    float acc[TM][TN]; // 寄存器用来作为累加器(计算TM×TN个C元素)
    float a_reg[TM];
    float b_reg[TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    int linear_tid = local_row * blockDim.x + local_col;
    int num_threads = blockDim.x * blockDim.y;

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        // 装载数据到共享内存
        int a_base_row = block_row;
        int a_base_col = k0;

        for (int idx = linear_tid; idx < BM * BK; idx += num_threads)
        {
            int r = idx / BK;
            int c = idx % BK;
            int a_row = a_base_row + r;
            int a_col = a_base_col + c;
            float v = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
            A_tile_t[c][r] = v;
        }

        int b_base_row = k0;
        int b_base_col = block_col;

        for (int idx = linear_tid; idx < BK * BN; idx += num_threads)
        {
            int r = idx / BN;
            int c = idx % BN;
            int b_row = b_base_row + r;
            int b_col = b_base_col + c;
            B_tile[r][c] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }

        __syncthreads();

        for (int t = 0; t < BK; ++t)
        {

#pragma unroll
            for (int i = 0; i < TM; ++i)
                a_reg[i] = A_tile_t[t][local_row * TM + i];

#pragma unroll
            for (int j = 0; j < TN; ++j)
                b_reg[j] = B_tile[t][local_col * TN + j];

#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
    {
        int row = row_base + i;
        if (row >= M)
            continue;

        for (int j = 0; j < TN; ++j)
        {
            int col = col_base + j;
            if (col >= N)
                continue;

            int out = row * N + col;
            if (beta == 0.0f)
            {
                C[out] = acc[i][j] * alpha;
            }
            else
            {
                C[out] = acc[i][j] * alpha + beta * C[out];
            }
        }
    }
}

piState piCudaGemmFp32_v3(float *__restrict__ A,
                          float *__restrict__ B,
                          float *__restrict__ C,
                          float alpha,
                          float beta,
                          int M, int K, int N)
{
    // C=αAB+βC
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int TM = 4;
    constexpr int TN = 4;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);

    gemm_kernel_v3<<<grid, block>>>(M, K, N, A, B, C, alpha, beta);
    return piSuccess;
}
