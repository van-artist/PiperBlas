#include <stdio.h>
#include <cuda_runtime.h>
#include "utils.h"

__global__ void helloFromGPU()
{
    printf("Hello from GPU!\n");
}
int main()
{
    printf("Hello, World!\n");
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    print_cuda_important_attrs(0);
    return 0;
}