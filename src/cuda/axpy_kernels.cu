#include <cuda_runtime.h>

__global__ void axpy_kernel(int n, int *A, int *B, int *C)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n)
    {
        C[index] = A[index] + B[index];
    }
}
