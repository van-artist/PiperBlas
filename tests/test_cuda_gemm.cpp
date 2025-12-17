#if !defined(PIPER_HAVE_CUDA) || !PIPER_HAVE_CUDA
#include <cstdio>

int main(int argc, char **argv)
{
    std::fprintf(stderr, "CUDA support is disabled.\n");
    return 0;
}

#else
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstring>

#include "core/common.h"
#include "cuda/cuda_kernels.h"

template <typename T>
static void fill_value(std::vector<T> &buf, T v)
{
    for (auto &x : buf)
        x = v;
}

template <typename T>
static float time_avg_ms(int warmup, int iters, auto launch)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i)
        launch();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        launch();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / (float)iters;
}

struct Result
{
    int N;
    float ms_fp32;
    float ms_fp64;
    double gflops_fp32;
    double gflops_fp64;
};

static Result run_case(int N)
{
    const int m = N, k = N, n = N;
    const size_t asz = (size_t)m * k;
    const size_t bsz = (size_t)k * n;
    const size_t csz = (size_t)m * n;
    const double ops = 2.0 * (double)m * (double)k * (double)n;

    std::vector<float> hA32(asz), hB32(bsz), hC32(csz);
    std::vector<double> hA64(asz), hB64(bsz), hC64(csz);
    fill_value(hA32, 1.0f);
    fill_value(hB32, 1.0f);
    fill_value(hC32, 0.0f);
    fill_value(hA64, 1.0);
    fill_value(hB64, 1.0);
    fill_value(hC64, 0.0);

    float *dA32 = nullptr, *dB32 = nullptr, *dC32 = nullptr;
    double *dA64 = nullptr, *dB64 = nullptr, *dC64 = nullptr;

    CHECK_CUDA(cudaMalloc(&dA32, asz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB32, bsz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC32, csz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dA64, asz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dB64, bsz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dC64, csz * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(dA32, hA32.data(), asz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB32, hB32.data(), bsz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC32, hC32.data(), csz * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(dA64, hA64.data(), asz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB64, hB64.data(), bsz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC64, hC64.data(), csz * sizeof(double), cudaMemcpyHostToDevice));

    auto ms32 = time_avg_ms<float>(2, 5, [&]()
                                   { (void)piCudaGemmFp32(dA32, dB32, dC32, 1.0f, 0.0f, m, k, n); });
    auto ms64 = time_avg_ms<double>(2, 5, [&]()
                                    { (void)piCudaGemmFp64(dA64, dB64, dC64, 1.0, 0.0, m, k, n); });

    CHECK_CUDA(cudaFree(dA32));
    CHECK_CUDA(cudaFree(dB32));
    CHECK_CUDA(cudaFree(dC32));
    CHECK_CUDA(cudaFree(dA64));
    CHECK_CUDA(cudaFree(dB64));
    CHECK_CUDA(cudaFree(dC64));

    Result r{};
    r.N = N;
    r.ms_fp32 = ms32;
    r.ms_fp64 = ms64;
    r.gflops_fp32 = ops / (ms32 * 1e-3) * 1e-9;
    r.gflops_fp64 = ops / (ms64 * 1e-3) * 1e-9;
    return r;
}

int main(int argc, char **argv)
{
    bool rich = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--rich") == 0)
            rich = true;

    std::vector<int> Ns;
    if (!rich)
    {
        int def[] = {64, 128, 256, 512};
        Ns.assign(def, def + sizeof(def) / sizeof(def[0]));
    }
    else
    {
        int richN[] = {
            32, 48, 64, 96,
            128, 160, 192, 224, 256,
            320, 384, 448, 512,
            640, 768, 896, 1024,
            1280, 1536, 1792, 2048,
            2560, 3072, 3584, 4096,
            5120, 6144, 7168, 8192};
        Ns.assign(richN, richN + sizeof(richN) / sizeof(richN[0]));
    }

    printf("==== CUDA GEMM (timing only) ====\n");
    printf("%8s | %10s %10s | %10s %10s\n", "N", "fp32_ms", "fp32_GF/s", "fp64_ms", "fp64_GF/s");

    for (int N : Ns)
    {
        Result r = run_case(N);
        printf("%8d | %10.3f %10.3f | %10.3f %10.3f\n",
               r.N, r.ms_fp32, r.gflops_fp32, r.ms_fp64, r.gflops_fp64);
    }

    return 0;
}
#endif
