#include <cuda_runtime.h>
#include "pi_type.h"

template <typename T>
__global__ void gemm_kernel(T *__restrict__ A, T *__restrict__ B, T *__restrict__ C, T alpha, T beta, int M, int K, int N)
{
    // 总体ikj循环
    // 每个线程负责一趟j，也就是完整处理完B的一行的一次缩放然后累加完毕
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    // 暂定每线程处理C的一行
    if (index < M)
    {
        // 整体缩放C：βC
        for (int j = 0; j < N; ++j)
        {
            C[index * N + j] *= beta;
        }
        // 矩阵乘：αAB
        T *C_row = C + index * N;
        T *A_row = A + index * K;
        T tmp = 0;
        for (size_t k = 0; k < K; k++)
        {
            T *B_row = B + k * N;
            for (size_t j = 0; j < N; j++)
            {
                C_row[j] += A_row[k] * B_row[j] * alpha;
            }
        }
    }
}

template <typename T>
piState gemm_cuda_impl(T *__restrict__ A, T *__restrict__ B, T *__restrict__ C, T alpha, T beta, int M, int K, int N)
{
    // C=αAB+βC
    constexpr int BLOCK_SIZE = 256;
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gemm_kernel<T><<<GRID_SIZE, BLOCK_SIZE>>>(A, B, C, alpha, beta, M, K, N);
    return piSuccess;
}

piState gemm_fp32(float *__restrict__ A, float *__restrict__ B, float *__restrict__ C, float alpha, float beta, int M, int K, int N)
{
    return gemm_cuda_impl<float>(A, B, C, alpha, beta, M, K, N);
}

piState gemm_fp64(double *__restrict__ A, double *__restrict__ B, double *__restrict__ C, double alpha, double beta, int M, int K, int N)
{
    return gemm_cuda_impl<double>(A, B, C, alpha, beta, M, K, N);
}
