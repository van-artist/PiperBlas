#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#include "utils.h"
#include "cuda/cuda_kernels.h"

static float bench_gemm(const char *name,
                        piState (*fn)(float *, float *, float *, float, float, int, int, int),
                        float *dA, float *dB, float *dC,
                        float alpha, float beta,
                        int m, int k, int n,
                        int warmup, int iters)
{
    CHECK_CUDA(cudaMemset(dC, 0, (size_t)m * n * sizeof(float)));

    for (int i = 0; i < warmup; ++i)
    {
        (void)fn(dA, dB, dC, alpha, beta, m, k, n);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
    {
        (void)fn(dA, dB, dC, alpha, beta, m, k, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    float avg_ms = ms / iters;
    double ops = 2.0 * (double)m * (double)k * (double)n;
    double gflops = ops / (avg_ms * 1e-3) * 1e-9;

    std::cout << name << ": avg " << avg_ms << " ms, " << gflops << " GF/s" << std::endl;
    return avg_ms;
}

int main()
{
    int m = 4096;
    int k = 4096;
    int n = 4096;

    size_t sizeA = (size_t)m * k;
    size_t sizeB = (size_t)k * n;
    size_t sizeC = (size_t)m * n;

    float *hostA = (float *)malloc(sizeA * sizeof(float));
    float *hostB = (float *)malloc(sizeB * sizeof(float));
    float *hostC = (float *)malloc(16 * sizeof(float));

    for (size_t i = 0; i < sizeA; ++i)
        hostA[i] = 1.0f;
    for (size_t i = 0; i < sizeB; ++i)
        hostB[i] = 1.0f;

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, sizeB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, sizeC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hostA, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hostB, sizeB * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    float beta = 0.0f;

    int warmup = 0;
    int iters = 1;

    bench_gemm("shape2 (fp32_v3)", piCudaGemmFp32_v4, dA, dB, dC, alpha, beta, m, k, n, warmup, iters);

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hostC, dC, 16 * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "C[0..15]: ";
    for (int i = 0; i < 16; ++i)
        std::cout << hostC[i] << (i + 1 == 16 ? '\n' : ' ');

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
