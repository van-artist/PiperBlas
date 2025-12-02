#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>
#include "utils.h"
#include "cuda/cuda_kernels.h"

__global__ void helloFromGPU()
{
    printf("Hello from GPU!\n");
}
int main()
{
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    print_cuda_important_attrs(0);
    return 0;
}