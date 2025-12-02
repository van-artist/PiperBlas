#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "pi_blas.h"
#include "pi_type.h"
#include "pi_config.h"
#include "utils.h"

typedef struct
{
    size_t N;
    size_t elems;
    double wall[3];
    double cpu[3];
    double gflops[3];
    double blas_wall;
    double blas_cpu;
    double blas_gflops;
    double cu_wall;
    double cu_cpu;
    double cu_gflops;
    double max_err[3];
    double l2[3];
    double max_err_cu;
    double l2_cu;
} Row;

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
        fprintf(stderr, "alloc fail\n");
        exit(1);
    }
    return p;
}

static void fill(double *x, size_t n, uint64_t *seed)
{
    for (size_t i = 0; i < n; i++)
    {
        *seed = (*seed * 2862933555777941757ULL) + 3037000493ULL;
        x[i] = ((double)(*seed >> 11) / (double)(1ULL << 53)) * 2.0 - 1.0;
    }
}

int main()
{
    config_init();

    const size_t Ns[] = {64, 128, 256, 512, 1024, 2048, 4096};
    const size_t n_scales = sizeof(Ns) / sizeof(Ns[0]);
    const double alpha = 1.2;
    const double beta = 0.8;

    Row rows[n_scales];

    piState (*funcs[3])(double *, double *, double *, double, double, size_t, size_t, size_t) =
        {piGemm, piGemm_v2, piGemm_v3};

    cublasHandle_t handle;
    cublasCreate(&handle);

    for (size_t t = 0; t < n_scales; t++)
    {
        size_t N = Ns[t];
        size_t m = N, k = N, n = N;
        size_t asz = m * k;
        size_t bsz = k * n;
        size_t csz = m * n;

        double *A = (double *)xalloc(asz * sizeof(double));
        double *B = (double *)xalloc(bsz * sizeof(double));
        double *Cref = (double *)xalloc(csz * sizeof(double));
        double *Cblas = (double *)xalloc(csz * sizeof(double));

        uint64_t seed = 1;
        fill(A, asz, &seed);
        fill(B, bsz, &seed);
        fill(Cref, csz, &seed);

        for (size_t i = 0; i < csz; i++)
            Cblas[i] = Cref[i];

        double w0 = wall_now();
        double c0 = cpu_now();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)m, (int)n, (int)k,
                    alpha, A, (int)k, B, (int)n, beta, Cblas, (int)n);
        double c1 = cpu_now();
        double w1 = wall_now();

        double blas_wall = w1 - w0;
        double blas_cpu = c1 - c0;
        double ops = 2.0 * (double)m * (double)k * (double)n;
        double blas_gflops = ops / blas_wall * 1e-9;

        double *Ccu = (double *)xalloc(csz * sizeof(double));
        for (size_t i = 0; i < csz; i++)
            Ccu[i] = Cref[i];

        double *dA = NULL;
        double *dB = NULL;
        double *dC = NULL;
        cudaMalloc(&dA, asz * sizeof(double));
        cudaMalloc(&dB, bsz * sizeof(double));
        cudaMalloc(&dC, csz * sizeof(double));

        cudaMemcpy(dA, A, asz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, bsz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dC, Ccu, csz * sizeof(double), cudaMemcpyHostToDevice);

        double w2 = wall_now();
        double c2 = cpu_now();
        cublasDgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)n, (int)m, (int)k,
                    &alpha,
                    dB, (int)n,
                    dA, (int)k,
                    &beta,
                    dC, (int)n);
        cudaDeviceSynchronize();
        double c3 = cpu_now();
        double w3 = wall_now();

        cudaMemcpy(Ccu, dC, csz * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        double cu_wall = w3 - w2;
        double cu_cpu = c3 - c2;
        double cu_gflops = ops / cu_wall * 1e-9;

        double max_err_cu = 0.0;
        double l2_cu = 0.0;

        for (size_t i = 0; i < csz; i++)
        {
            double d = fabs(Ccu[i] - Cblas[i]);
            if (d > max_err_cu)
                max_err_cu = d;
            l2_cu += d * d;
        }
        l2_cu = sqrt(l2_cu);

        for (int v = 0; v < 3; v++)
        {
            double *C = (double *)xalloc(csz * sizeof(double));
            for (size_t i = 0; i < csz; i++)
                C[i] = Cref[i];

            double wv0 = wall_now();
            double cv0 = cpu_now();
            funcs[v](A, B, C, alpha, beta, m, k, n);
            double cv1 = cpu_now();
            double wv1 = wall_now();

            rows[t].wall[v] = wv1 - wv0;
            rows[t].cpu[v] = cv1 - cv0;
            rows[t].gflops[v] = ops / rows[t].wall[v] * 1e-9;

            double max_err = 0.0;
            double l2 = 0.0;
            for (size_t i = 0; i < csz; i++)
            {
                double d = fabs(C[i] - Cblas[i]);
                if (d > max_err)
                    max_err = d;
                l2 += d * d;
            }
            rows[t].max_err[v] = max_err;
            rows[t].l2[v] = sqrt(l2);

            free(C);
        }

        rows[t].N = N;
        rows[t].elems = csz;
        rows[t].blas_wall = blas_wall;
        rows[t].blas_cpu = blas_cpu;
        rows[t].blas_gflops = blas_gflops;
        rows[t].cu_wall = cu_wall;
        rows[t].cu_cpu = cu_cpu;
        rows[t].cu_gflops = cu_gflops;
        rows[t].max_err_cu = max_err_cu;
        rows[t].l2_cu = l2_cu;

        free(A);
        free(B);
        free(Cref);
        free(Cblas);
        free(Ccu);
    }

    cublasDestroy(handle);

    printf("%8s %8s | %10s %10s %10s | %10s %10s | %10s %10s %10s %10s\n",
           "N", "elems",
           "v1_GF/s", "v2_GF/s", "v3_GF/s",
           "blas_GF/s", "cu_GF/s",
           "v1_err", "v2_err", "v3_err", "cu_err");

    for (size_t i = 0; i < n_scales; i++)
    {
        printf("%8zu %8zu | %10.3f %10.3f %10.3f | %10.3f %10.3f | %10.3e %10.3e %10.3e %10.3e\n",
               rows[i].N, rows[i].elems,
               rows[i].gflops[0], rows[i].gflops[1], rows[i].gflops[2],
               rows[i].blas_gflops, rows[i].cu_gflops,
               rows[i].max_err[0], rows[i].max_err[1], rows[i].max_err[2],
               rows[i].max_err_cu);
    }

    return 0;
}
