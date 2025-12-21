#include <cuda_runtime.h>
#include "pi_type.hpp"

__global__ void axpy_kernel(int n, float alpha, const float *__restrict__ x, float *__restrict__ y)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
    {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

piState pi_cuda_axpy(int n, float alpha, const float *__restrict__ x, float *__restrict__ y)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    axpy_kernel<<<numBlocks, blockSize>>>(n, alpha, x, y);
    return piSuccess;
}
