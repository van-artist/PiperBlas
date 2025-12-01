#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU()
{
    printf("Hello from GPU!\n");
}
int main()
{
    printf("Hello, World!\n");
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}