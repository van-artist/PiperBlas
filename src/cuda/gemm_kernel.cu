#include <cuda_runtime.h>
#include "pi_type.h"

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
        // #pragma unroll
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

    constexpr int Mt = 16; // M方向的块长度
    constexpr int Kt = 16; // K方向的块长度
    constexpr int Nt = 16; // N方向的块长度

    // shared memory
    __shared__ T A_tile[Mt][Kt];
    __shared__ T B_tile[Kt][Nt];

    // 明确索引
    int c_tile_row = blockIdx.y; // Ctile在所有分块中的行索引
    int c_tile_col = blockIdx.x; // Ctile在所有分块中的列索引
    int local_row = threadIdx.y; // 线程负责Ctile元素的行索引
    int local_col = threadIdx.x; // 线程负责Ctile元素的列索引

    int row = c_tile_row * Mt + local_row; // 线程负责元素在C中的行索引
    int col = c_tile_col * Nt + local_col; // 线程负责元素在C中的列索引

    T acc = 0;
    for (int k = 0; k < K; k += Kt)
    {
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
        for (int t = 0; t < Kt; ++t)
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
