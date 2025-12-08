#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>
#include <cstdlib>

#include "utils.h"
#include "cuda/cuda_kernels.h"

int main()
{
    // 8192 x 8192 GEMM: C = alpha * A * B + beta * C
    int m = 8192;
    int k = 8192;
    int n = 8192;

    size_t sizeA = static_cast<size_t>(m) * k;
    size_t sizeB = static_cast<size_t>(k) * n;
    size_t sizeC = static_cast<size_t>(m) * n;

    float *hostA = (float *)malloc(sizeA * sizeof(float));
    float *hostB = (float *)malloc(sizeB * sizeof(float));
    float *hostC = (float *)malloc(sizeC * sizeof(float));

    // 简单初始化一下数据
    for (size_t i = 0; i < sizeA; ++i)
        hostA[i] = 1.0f;
    for (size_t i = 0; i < sizeB; ++i)
        hostB[i] = 1.0f;
    for (size_t i = 0; i < sizeC; ++i)
        hostC[i] = 0.0f;

    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, sizeA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, sizeB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, sizeC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hostA, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hostB, sizeB * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hostC, sizeC * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    float beta = 0.0f;
    piState state = piCudaGemmFp32_v2(dA, dB, dC, alpha, beta, m, k, n);

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hostC, dC, 16 * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "gemm state: " << state << std::endl;
    for (int i = 0; i < 16; ++i)
    {
        std::cout << hostC[i] << (i + 1 == 16 ? '\n' : ' ');
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
