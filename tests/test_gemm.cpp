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
    size_t N, elems;
    double pi_wall, blas_wall, cu_wall;
    double pi_cpu, blas_cpu;
    double pi_gflops, blas_gflops, cu_gflops;
    double max_err_pi, max_err_cu;
    double l2_pi, l2_cu;
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

static void *xalloc(size_t n)
{
    void *p = NULL;
    posix_memalign(&p, 64, n);
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
    const double alpha = 1.2, beta = 0.8;

    Row rows[n_scales];
    cublasHandle_t handle;
    cublasCreate(&handle);

    for (size_t t = 0; t < n_scales; t++)
    {
        size_t N = Ns[t], m = N, k = N, n = N;
        size_t asz = m * k, bsz = k * n, csz = m * n;

        double *A = (double *)xalloc(asz * sizeof(double));
        double *B = (double *)xalloc(bsz * sizeof(double));
        double *C1 = (double *)xalloc(csz * sizeof(double));
        double *C2 = (double *)xalloc(csz * sizeof(double));
        double *C3 = (double *)xalloc(csz * sizeof(double));

        uint64_t seed = 1;
        fill(A, asz, &seed);
        fill(B, bsz, &seed);
        fill(C1, csz, &seed);
        for (size_t i = 0; i < csz; i++)
        {
            C2[i] = C1[i];
            C3[i] = C1[i];
        }

        double w0 = wall_now(), c0 = cpu_now();
        piGemm_v2(A, B, C1, alpha, beta, m, k, n);
        double c1 = cpu_now(), w1 = wall_now();

        double w2 = wall_now(), c2 = cpu_now();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, k, B, n, beta, C2, n);
        double c3 = cpu_now(), w3 = wall_now();

        double *dA, *dB, *dC;
        cudaMalloc(&dA, asz * sizeof(double));
        cudaMalloc(&dB, bsz * sizeof(double));
        cudaMalloc(&dC, csz * sizeof(double));

        cudaMemcpy(dA, A, asz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, bsz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dC, C3, csz * sizeof(double), cudaMemcpyHostToDevice);

        double w4 = wall_now(), c4 = cpu_now();
        cublasDgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    dB, n,
                    dA, k,
                    &beta,
                    dC, n);
        cudaDeviceSynchronize();
        double c5 = cpu_now(), w5 = wall_now();

        cudaMemcpy(C3, dC, csz * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        double max_err_pi = 0, max_err_cu = 0;
        double l2_pi = 0, l2_cu = 0;

        for (size_t i = 0; i < csz; i++)
        {
            double d1 = fabs(C1[i] - C2[i]);
            double d2 = fabs(C3[i] - C2[i]);
            if (d1 > max_err_pi)
                max_err_pi = d1;
            if (d2 > max_err_cu)
                max_err_cu = d2;
            l2_pi += d1 * d1;
            l2_cu += d2 * d2;
        }

        l2_pi = sqrt(l2_pi);
        l2_cu = sqrt(l2_cu);

        double ops = 2.0 * m * k * n;
        double pi_wall = w1 - w0;
        double blas_wall = w3 - w2;
        double cu_wall = w5 - w4;

        double gflops_pi = ops / pi_wall * 1e-9;
        double gflops_blas = ops / blas_wall * 1e-9;
        double gflops_cu = ops / cu_wall * 1e-9;

        rows[t] = (Row){
            N, csz,
            pi_wall, blas_wall, cu_wall,
            c1 - c0, c3 - c2,
            gflops_pi, gflops_blas, gflops_cu,
            max_err_pi, max_err_cu,
            l2_pi, l2_cu};

        free(A);
        free(B);
        free(C1);
        free(C2);
        free(C3);
    }

    cublasDestroy(handle);

    printf("%8s %12s %12s %12s %12s %12s %12s %12s\n",
           "N", "elems", "pi_GF/s", "blas_GF/s", "cu_GF/s",
           "max_pi", "max_cu", " ");

    for (size_t i = 0; i < n_scales; i++)
    {
        printf("%8zu %12zu %12.3f %12.3f %12.3f %12.3e %12.3e\n",
               rows[i].N, rows[i].elems,
               rows[i].pi_gflops, rows[i].blas_gflops, rows[i].cu_gflops,
               rows[i].max_err_pi, rows[i].max_err_cu);
    }

    return 0;
}
