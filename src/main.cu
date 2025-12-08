#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>
#include <cstdlib>

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
    int m = 2;
    int k = 2;
    int n = 2;
    float hostA[] = {1.f, 2.f, 3.f, 4.f};
    float hostB[] = {5.f, 6.f, 7.f, 8.f};
    float hostC[] = {1.f, 1.f, 1.f, 1.f};
    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(hostA)));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(hostB)));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(hostC)));
    CHECK_CUDA(cudaMemcpy(dA, hostA, sizeof(hostA), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hostB, sizeof(hostB), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hostC, sizeof(hostC), cudaMemcpyHostToDevice));
    piState state = piCudaGemmFp32_v2(dA, dB, dC, 1.0f, 2.0f, m, k, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hostC, dC, sizeof(hostC), cudaMemcpyDeviceToHost));
    std::cout << "gemm state: " << state << std::endl;
    for (int i = 0; i < m * n; ++i)
    {
        std::cout << hostC[i] << (i + 1 == m * n ? '\n' : ' ');
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
