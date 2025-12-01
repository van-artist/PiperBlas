#pragma once
#include <cuda_runtime.h>
__global__ void axpy_kernel(int n, int *A, int *B, int *C);