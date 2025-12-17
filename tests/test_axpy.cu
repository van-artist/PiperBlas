#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "cuda/cuda_kernels.h"

int main()
{
    const int Ns[] = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18};
    const int iters = 100;
    const float alpha = 2.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    std::cout << std::setw(12) << "N"
              << std::setw(12) << "my_ms"
              << std::setw(12) << "my_GB/s"
              << std::setw(12) << "cu_ms"
              << std::setw(12) << "cu_GB/s"
              << std::setw(12) << "max_err"
              << std::setw(12) << "l2_err"
              << std::setw(14) << "ratio"
              << "\n";

    for (int n : Ns)
    {
        std::vector<float> hx(n), hy0(n), hy_my(n), hy_cu(n), href(n);
        for (int i = 0; i < n; ++i)
        {
            hx[i] = i * 0.001f;
            hy0[i] = i * 0.002f;
        }

        float *dx, *dy_my, *dy_cu;
        cudaMalloc(&dx, n * sizeof(float));
        cudaMalloc(&dy_my, n * sizeof(float));
        cudaMalloc(&dy_cu, n * sizeof(float));

        cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dy_my, hy0.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t s1, e1;
        cudaEventCreate(&s1);
        cudaEventCreate(&e1);

        cudaEventRecord(s1);
        for (int i = 0; i < iters; ++i)
            pi_cuda_axpy(n, alpha, dx, dy_my);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);

        float my_ms;
        cudaEventElapsedTime(&my_ms, s1, e1);
        my_ms /= iters;

        cudaMemcpy(hy_my.data(), dy_my, n * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < n; ++i)
            href[i] = hy0[i] + alpha * hx[i] * iters;

        double max_err = 0.0, l2_err = 0.0;
        for (int i = 0; i < n; ++i)
        {
            double d = hy_my[i] - href[i];
            if (std::abs(d) > max_err)
                max_err = std::abs(d);
            l2_err += d * d;
        }
        l2_err = std::sqrt(l2_err);

        cudaMemcpy(dy_cu, hy0.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaEvent_t s2, e2;
        cudaEventCreate(&s2);
        cudaEventCreate(&e2);

        cudaEventRecord(s2);
        for (int i = 0; i < iters; ++i)
            cublasSaxpy(handle, n, &alpha, dx, 1, dy_cu, 1);
        cudaEventRecord(e2);
        cudaEventSynchronize(e2);

        float cu_ms;
        cudaEventElapsedTime(&cu_ms, s2, e2);
        cu_ms /= iters;

        cudaMemcpy(hy_cu.data(), dy_cu, n * sizeof(float), cudaMemcpyDeviceToHost);

        double bytes = 3.0 * n * sizeof(float);
        double my_gbps = bytes / (my_ms * 1e-3) / 1e9;
        double cu_gbps = bytes / (cu_ms * 1e-3) / 1e9;
        double ratio = my_gbps / cu_gbps;

        std::cout << std::setw(12) << n
                  << std::setw(12) << std::fixed << std::setprecision(4) << my_ms
                  << std::setw(12) << std::fixed << std::setprecision(3) << my_gbps
                  << std::setw(12) << std::fixed << std::setprecision(4) << cu_ms
                  << std::setw(12) << std::fixed << std::setprecision(3) << cu_gbps
                  << std::setw(12) << std::fixed << std::setprecision(3) << max_err
                  << std::setw(12) << std::fixed << std::setprecision(3) << l2_err
                  << std::setw(14) << std::fixed << std::setprecision(3) << ratio
                  << "\n";

        cudaFree(dx);
        cudaFree(dy_my);
        cudaFree(dy_cu);
    }

    cublasDestroy(handle);
    return 0;
}
