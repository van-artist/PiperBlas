#include <cuda_runtime.h>
#include "pi_type.h"
#include "utils.h"

__global__ void axpy_kernel(int n, float alpha, const float *__restrict__ x, float *__restrict__ y)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
    {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

piState axpy(int n, float alpha, const float *__restrict__ x, float *__restrict__ y)
{
    constexpr int BLOCK_DIM = 256;
    int GRID_DIM = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    axpy_kernel<<<GRID_DIM, BLOCK_DIM>>>(n, alpha, x, y);
    return piSuccess;
}