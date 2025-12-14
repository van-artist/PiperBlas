#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <string>
#include <string.h>

#include "pi_blas.h"
#include "pi_type.h"
#include "pi_config.h"
#include "utils.h"
#include "cuda/cuda_kernels.h"

piState piCudaGemmFp32_v4(float *__restrict__ A,
                          float *__restrict__ B,
                          float *__restrict__ C,
                          float alpha,
                          float beta,
                          int M, int K, int N);

static void *aligned_alloc64(size_t bytes)
{
    void *p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0 || !p)
    {
        fprintf(stderr, "alloc fail\n");
        exit(1);
    }
    return p;
}

static void fill_random(float *x, size_t n, uint64_t *seed)
{
    for (size_t i = 0; i < n; i++)
    {
        *seed = (*seed * 2862933555777941757ULL) + 3037000493ULL;
        double v = ((double)(*seed >> 11) / (double)(1ULL << 53)) * 2.0 - 1.0;
        x[i] = (float)v;
    }
}

static void host_gemm_rowmajor(int m, int n, int k,
                               float alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               float beta,
                               float *C, int ldc)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                alpha, A, lda,
                B, ldb,
                beta, C, ldc);
}

static void device_gemm_cublas_rowmajor(cublasHandle_t handle,
                                        int m, int n, int k,
                                        float alpha,
                                        const float *dA, int lda,
                                        const float *dB, int ldb,
                                        float beta,
                                        float *dC, int ldc)
{
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                dB, ldb,
                dA, lda,
                &beta,
                dC, ldc);
}

template <class F>
static float time_avg_ms(F launch, int warmup, int iters)
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

static double max_abs_error(const float *x, const float *ref, size_t n)
{
    double e = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double d = (double)x[i] - (double)ref[i];
        double a = fabs(d);
        if (a > e)
            e = a;
    }
    return e;
}

struct Result
{
    size_t N;
    double gflops_custom1, gflops_custom2, gflops_custom3, gflops_custom4, gflops_cublas;
    double err_custom1, err_custom2, err_custom3, err_custom4, err_cublas;
};

static Result run_case(cublasHandle_t handle, size_t N)
{
    const int m = (int)N, n = (int)N, k = (int)N;
    const size_t asz = (size_t)m * k;
    const size_t bsz = (size_t)k * n;
    const size_t csz = (size_t)m * n;

    const float alpha = 1.2f;
    const float beta_ref = 0.8f;
    const float beta_time = 0.0f;

    float *A = (float *)aligned_alloc64(asz * sizeof(float));
    float *B = (float *)aligned_alloc64(bsz * sizeof(float));
    float *C0 = (float *)aligned_alloc64(csz * sizeof(float));
    float *Cref = (float *)aligned_alloc64(csz * sizeof(float));

    uint64_t seed = 1;
    fill_random(A, asz, &seed);
    fill_random(B, bsz, &seed);
    fill_random(C0, csz, &seed);

    for (size_t i = 0; i < csz; ++i)
        Cref[i] = C0[i];

    host_gemm_rowmajor(m, n, k,
                       alpha,
                       A, k,
                       B, n,
                       beta_ref,
                       Cref, n);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, asz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, bsz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, csz * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, A, asz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, bsz * sizeof(float), cudaMemcpyHostToDevice));

    float *C = (float *)aligned_alloc64(csz * sizeof(float));
    const double ops = 2.0 * (double)m * (double)n * (double)k;

    const int WARMUP = 2;
    const int ITERS = 3;

    Result out{};
    out.N = N;

    auto time_only = [&](auto launch, double &gflops)
    {
        CHECK_CUDA(cudaMemset(dC, 0, csz * sizeof(float)));
        float ms = time_avg_ms(launch, WARMUP, ITERS);
        gflops = ops / (ms * 1e-3) * 1e-9;
    };

    auto check_only = [&](auto launch, double &err)
    {
        CHECK_CUDA(cudaMemcpy(dC, C0, csz * sizeof(float), cudaMemcpyHostToDevice));
        launch();
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(C, dC, csz * sizeof(float), cudaMemcpyDeviceToHost));
        err = max_abs_error(C, Cref, csz);
    };

    time_only([&]()
              { device_gemm_cublas_rowmajor(handle, m, n, k, alpha, dA, k, dB, n, beta_time, dC, n); },
              out.gflops_cublas);
    check_only([&]()
               { device_gemm_cublas_rowmajor(handle, m, n, k, alpha, dA, k, dB, n, beta_ref, dC, n); },
               out.err_cublas);

    time_only([&]()
              { (void)piCudaGemmFp32(dA, dB, dC, alpha, beta_time, m, k, n); },
              out.gflops_custom1);
    check_only([&]()
               { (void)piCudaGemmFp32(dA, dB, dC, alpha, beta_ref, m, k, n); },
               out.err_custom1);

    time_only([&]()
              { (void)piCudaGemmFp32_v2(dA, dB, dC, alpha, beta_time, m, k, n); },
              out.gflops_custom2);
    check_only([&]()
               { (void)piCudaGemmFp32_v2(dA, dB, dC, alpha, beta_ref, m, k, n); },
               out.err_custom2);

    time_only([&]()
              { (void)piCudaGemmFp32_v3(dA, dB, dC, alpha, beta_time, m, k, n); },
              out.gflops_custom3);
    check_only([&]()
               { (void)piCudaGemmFp32_v3(dA, dB, dC, alpha, beta_ref, m, k, n); },
               out.err_custom3);

    time_only([&]()
              { (void)piCudaGemmFp32_v4(dA, dB, dC, alpha, beta_time, m, k, n); },
              out.gflops_custom4);
    check_only([&]()
               { (void)piCudaGemmFp32_v4(dA, dB, dC, alpha, beta_ref, m, k, n); },
               out.err_custom4);

    free(A);
    free(B);
    free(C0);
    free(Cref);
    free(C);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return out;
}

enum ShowMask : int
{
    SHOW_ABS = 1 << 0,
    SHOW_REL = 1 << 1,
    SHOW_ERR = 1 << 2,
};

static int parse_show_mask(int argc, char **argv)
{
    int mask = SHOW_ABS | SHOW_REL;
    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        if (strncmp(a, "--show=", 7) == 0)
        {
            mask = 0;
            std::string s(a + 7);
            for (auto &c : s)
                if (c == ',')
                    c = '+';
            if (s.find("abs") != std::string::npos)
                mask |= SHOW_ABS;
            if (s.find("rel") != std::string::npos)
                mask |= SHOW_REL;
            if (s.find("err") != std::string::npos)
                mask |= SHOW_ERR;
            if (mask == 0)
                mask = SHOW_ABS | SHOW_REL;
        }
    }
    return mask;
}

static bool use_rich_shapes(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "--rich") == 0)
            return true;
    return false;
}

static inline double rel_pct(double custom, double cublas)
{
    return cublas > 0.0 ? (custom / cublas) * 100.0 : 0.0;
}

int main(int argc, char **argv)
{
    config_init();

    int show = parse_show_mask(argc, argv);
    bool rich = use_rich_shapes(argc, argv);

    cublasHandle_t handle;
    cublasCreate(&handle);

    std::vector<size_t> Ns;

    if (!rich)
    {
        size_t def[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
        Ns.assign(def, def + sizeof(def) / sizeof(def[0]));
    }
    else
    {
        size_t richN[] = {
            32, 48, 64, 96,
            128, 160, 192, 224, 256,
            320, 384, 448, 512,
            640, 768, 896, 1024,
            1280, 1536, 1792, 2048,
            2560, 3072, 3584, 4096,
            5120, 6144, 7168, 8192};
        Ns.assign(richN, richN + sizeof(richN) / sizeof(richN[0]));
    }

    std::vector<Result> results;
    for (auto N : Ns)
        results.push_back(run_case(handle, N));

    printf("==== CUDA fp32 ====\n");

    printf("%8s", "N");
    if (show & SHOW_ABS)
        printf(" | %10s %10s %10s %10s %10s", "c1_GF/s", "c2_GF/s", "c3_GF/s", "c4_GF/s", "cu_GF/s");
    if (show & SHOW_REL)
        printf(" | %10s %10s %10s %10s", "c1_%cu", "c2_%cu", "c3_%cu", "c4_%cu");
    if (show & SHOW_ERR)
        printf(" | %10s %10s %10s %10s %10s", "c1_err", "c2_err", "c3_err", "c4_err", "cu_err");
    printf("\n");

    for (const auto &r : results)
    {
        printf("%8zu", r.N);

        if (show & SHOW_ABS)
            printf(" | %10.3f %10.3f %10.3f %10.3f %10.3f",
                   r.gflops_custom1, r.gflops_custom2, r.gflops_custom3, r.gflops_custom4, r.gflops_cublas);

        if (show & SHOW_REL)
            printf(" | %10.2f %10.2f %10.2f %10.2f",
                   rel_pct(r.gflops_custom1, r.gflops_cublas),
                   rel_pct(r.gflops_custom2, r.gflops_cublas),
                   rel_pct(r.gflops_custom3, r.gflops_cublas),
                   rel_pct(r.gflops_custom4, r.gflops_cublas));

        if (show & SHOW_ERR)
            printf(" | %10.3e %10.3e %10.3e %10.3e %10.3e",
                   r.err_custom1, r.err_custom2, r.err_custom3, r.err_custom4, r.err_cublas);

        printf("\n");
    }

    cublasDestroy(handle);
    return 0;
}
