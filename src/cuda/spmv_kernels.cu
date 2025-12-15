#include <cuda_runtime.h>
#include "utils.h"
#include "pi_type.h"
#include "pi_csr.h"

template <typename T>
__global__ void spmv_kernel(pi_csr A, const double *__restrict x, double *__restrict y)
{
    // 每个线程负责一个y中的元素，对应A的一行和x点积
    int row_idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (row_idx >= A.n_rows)
        return;
    T acc = (T)0;
    int start = A.row_ptr[row_idx];
    int end = A.row_ptr[row_idx + 1];
    for (int j = start; j < end; ++j)
        acc += (T)A.values[j] * (T)x[A.col_idx[j]];
    y[row_idx] = (double)acc;
}

template <typename T>
piState pi_cuda_spmv_impl(const pi_csr *__restrict A, double *__restrict x, double *__restrict y)
{
    // y=A*x
    constexpr int BLOCK_SIZE = 256;
    // 每个block有8个warp
    int grid = (A->n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if constexpr (std::is_same_v<T, double>)
        spmv_kernel_fp64<<<grid, BLOCK_SIZE>>>(*A, x, y);
    else
        spmv_kernel_fp32<<<grid, BLOCK_SIZE>>>(*A, x, y);

    CHECK_CUDA(cudaGetLastError());
    return piSuccess;
}

piState pi_cuda_spmv_fp32(const pi_csr *__restrict A, double *__restrict x, double *__restrict y)
{
    return pi_cuda_spmv_impl<float>(A, x, y);
}

piState pi_cuda_spmv_fp64(const pi_csr *__restrict A, double *__restrict x, double *__restrict y)
{
    return pi_cuda_spmv_impl<double>(A, x, y);
}