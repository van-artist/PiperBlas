#include <cstdio>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int M = 4, K = 3, N = 5;

    std::vector<float> hA(M * K), hB(K * N), hC(M * N);
    for (int i = 0; i < M * K; i++)
        hA[i] = (i % 7 + 1);
    for (int i = 0; i < K * N; i++)
        hB[i] = (i % 5 + 1);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    // 1. 看 CUDA 自己有没有报错
    cudaError_t e0 = cudaSetDevice(0);
    std::cout << "cudaSetDevice(0) err=" << (int)e0
              << " (" << cudaGetErrorString(e0) << ")\n";

    cudaError_t e1 = cudaMalloc(&dA, M * K * sizeof(float));
    cudaError_t e2 = cudaMalloc(&dB, K * N * sizeof(float));
    cudaError_t e3 = cudaMalloc(&dC, M * N * sizeof(float));
    std::cout << "cudaMalloc dA/dB/dC err="
              << (int)e1 << "," << (int)e2 << "," << (int)e3 << "\n";

    cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 给 C 一个明显不是 0 的初值，方便看 beta 的效果
    for (int i = 0; i < M * N; ++i)
        hC[i] = 123.0f;
    cudaMemcpy(dC, hC.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;

    cublasHandle_t handle;
    cublasStatus_t st = cublasCreate(&handle);
    std::cout << "cublasCreate status=" << (int)st << "\n";
    if (st != CUBLAS_STATUS_SUCCESS)
    {
        return 1;
    }

    cublasStatus_t st_gemm = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        dA, M,
        dB, K,
        &beta,
        dC, M);

    std::cout << "cublasSgemm status=" << (int)st_gemm << "\n";

    // 同步一下，看 runtime 这边有没有察觉到错误
    cudaError_t e_sync = cudaDeviceSynchronize();
    std::cout << "cudaDeviceSynchronize err=" << (int)e_sync
              << " (" << cudaGetErrorString(e_sync) << ")\n";

    cudaMemcpy(hC.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M * N; i++)
        printf("%.1f ", hC[i]);
    printf("\n");

    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
