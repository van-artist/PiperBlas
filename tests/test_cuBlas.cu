#include <cblas.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include <string.h>

static double wall_now()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static double cpu_now()
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void *xalloc(size_t n_bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n_bytes) != 0 || !p)
    {
        std::fprintf(stderr, "alloc fail\n");
        std::exit(1);
    }
    return p;
}

static void fill(float *x, size_t n, uint64_t *seed)
{
    for (size_t i = 0; i < n; i++)
    {
        *seed = (*seed * 2862933555777941757ULL) + 3037000493ULL;
        x[i] = (float)(((double)(*seed >> 11) / (double)(1ULL << 53)) * 2.0 - 1.0);
    }
}

int main()
{
    const size_t Ns[] = {128, 256, 512, 1024, 2048, 4096, 8192};
    const size_t n_scales = sizeof(Ns) / sizeof(Ns[0]);
    const float alpha = 1.2f;
    const float beta = 0.8f;
    const int iters = 10;

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "cublasCreate failed\n");
        return 1;
    }

    std::printf("%8s %12s | %10s %10s | %10s %10s | %10s %10s\n",
                "N", "elems",
                "blas_ms", "blas_GF/s",
                "cu_ms", "cu_GF/s",
                "max_err", "l2_err");

    for (size_t t = 0; t < n_scales; t++)
    {
        size_t N = Ns[t];
        size_t m = N, k = N, n = N;
        size_t asz = m * k;
        size_t bsz = k * n;
        size_t csz = m * n;

        float *A = (float *)xalloc(asz * sizeof(float));
        float *B = (float *)xalloc(bsz * sizeof(float));
        float *C0 = (float *)xalloc(csz * sizeof(float));
        float *Ccpu = (float *)xalloc(csz * sizeof(float));
        float *Cgpu = (float *)xalloc(csz * sizeof(float));

        uint64_t seed = 1;
        fill(A, asz, &seed);
        fill(B, bsz, &seed);
        fill(C0, csz, &seed);

        memcpy(Ccpu, C0, csz * sizeof(float));

        double w0 = wall_now();
        double c0 = cpu_now();
        for (int it = 0; it < iters; ++it)
        {
            memcpy(Ccpu, C0, csz * sizeof(float));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)m, (int)n, (int)k,
                        alpha, A, (int)k,
                        B, (int)n,
                        beta, Ccpu, (int)n);
        }
        double c1 = cpu_now();
        double w1 = wall_now();

        double blas_wall = (w1 - w0) / iters;
        double blas_cpu = (c1 - c0) / iters;
        (void)blas_cpu;

        double ops = 2.0 * (double)m * (double)k * (double)n;
        double blas_gflops = ops / blas_wall * 1e-9;

        memcpy(Cgpu, C0, csz * sizeof(float));

        float *dA = nullptr;
        float *dB = nullptr;
        float *dC = nullptr;

        cudaError_t cerr;
        cerr = cudaMalloc(&dA, asz * sizeof(float));
        if (cerr != cudaSuccess)
        {
            std::fprintf(stderr, "cudaMalloc dA failed\n");
            return 1;
        }
        cerr = cudaMalloc(&dB, bsz * sizeof(float));
        if (cerr != cudaSuccess)
        {
            std::fprintf(stderr, "cudaMalloc dB failed\n");
            return 1;
        }
        cerr = cudaMalloc(&dC, csz * sizeof(float));
        if (cerr != cudaSuccess)
        {
            std::fprintf(stderr, "cudaMalloc dC failed\n");
            return 1;
        }

        cudaMemcpy(dA, A, asz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, bsz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dC, Cgpu, csz * sizeof(float), cudaMemcpyHostToDevice);

        double w2 = wall_now();
        double c2 = cpu_now();
        for (int it = 0; it < iters; ++it)
        {
            cublasStatus_t st = cublasSgemm(handle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            (int)n, (int)m, (int)k,
                                            &alpha,
                                            dB, (int)n,
                                            dA, (int)k,
                                            &beta,
                                            dC, (int)n);
            if (st != CUBLAS_STATUS_SUCCESS)
            {
                std::fprintf(stderr, "cublasSgemm failed\n");
                return 1;
            }
        }
        cudaDeviceSynchronize();
        double c3 = cpu_now();
        double w3 = wall_now();

        cudaMemcpy(Cgpu, dC, csz * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        double cu_wall = (w3 - w2) / iters;
        double cu_cpu = (c3 - c2) / iters;
        (void)cu_cpu;

        double cu_gflops = ops / cu_wall * 1e-9;

        double max_err = 0.0;
        double l2_err = 0.0;
        for (size_t i = 0; i < csz; ++i)
        {
            double d = (double)Cgpu[i] - (double)Ccpu[i];
            if (std::fabs(d) > max_err)
                max_err = std::fabs(d);
            l2_err += d * d;
        }
        l2_err = std::sqrt(l2_err);

        std::printf("%8zu %12zu | %10.4f %10.3f | %10.4f %10.3f | %10.3e %10.3e\n",
                    N, csz,
                    blas_wall * 1e3, blas_gflops,
                    cu_wall * 1e3, cu_gflops,
                    max_err, l2_err);

        free(A);
        free(B);
        free(C0);
        free(Ccpu);
        free(Cgpu);
    }

    cublasDestroy(handle);
    return 0;
}
