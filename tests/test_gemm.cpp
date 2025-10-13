#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
extern "C"
{
#include "pi_blas.h"
#include "pi_type.h"
#include "pi_config.h"
}

typedef struct
{
    size_t N, elems;
    double pi_wall, blas_wall;
    double pi_cpu, blas_cpu;
    double pi_gflops, blas_gflops;
    double max_err, l2;
} Row;

static double wall_now_gettimeofday()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static double cpu_now_clock()
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void *xalloc(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n) != 0 || !p)
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
    const size_t Ns[] = {64, 128, 256, 512, 1024, 2048};
    const size_t n_scales = sizeof(Ns) / sizeof(Ns[0]);
    const double alpha = 1.2, beta = 0.8;
    Row rows[sizeof(Ns) / sizeof(Ns[0])];

    for (size_t t = 0; t < n_scales; t++)
    {
        size_t N = Ns[t];
        size_t m = N, k = N, n = N;

        size_t asz = m * k, bsz = k * n, csz = m * n;
        double *A = (double *)xalloc(asz * sizeof(double));
        double *B = (double *)xalloc(bsz * sizeof(double));
        double *C1 = (double *)xalloc(csz * sizeof(double));
        double *C2 = (double *)xalloc(csz * sizeof(double));

        uint64_t seed = 1;
        fill(A, asz, &seed);
        fill(B, bsz, &seed);
        fill(C1, csz, &seed);
        for (size_t i = 0; i < csz; i++)
            C2[i] = C1[i];

        double w0 = wall_now_gettimeofday();
        double c0 = cpu_now_clock();
        piGemm(A, B, C1, alpha, beta, m, k, n);
        double c1 = cpu_now_clock();
        double w1 = wall_now_gettimeofday();

        double w2 = wall_now_gettimeofday();
        double c2 = cpu_now_clock();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)m, (int)n, (int)k,
                    alpha, A, (int)k, B, (int)n, beta, C2, (int)n);
        double c3 = cpu_now_clock();
        double w3 = wall_now_gettimeofday();

        double max_err = 0.0, l2 = 0.0;
        for (size_t i = 0; i < csz; i++)
        {
            double d = fabs(C1[i] - C2[i]);
            if (d > max_err)
                max_err = d;
            l2 += d * d;
        }
        l2 = sqrt(l2);

        double ops = 2.0 * (double)m * (double)k * (double)n;
        double pi_wall = w1 - w0, blas_wall = w3 - w2;
        double pi_cpu = c1 - c0, blas_cpu = c3 - c2;
        double gflops_pi = ops / pi_wall * 1e-9;
        double gflops_blas = ops / blas_wall * 1e-9;

        rows[t] = (Row){N, csz, pi_wall, blas_wall, pi_cpu, blas_cpu, gflops_pi, gflops_blas, max_err, l2};

        free(A);
        free(B);
        free(C1);
        free(C2);
    }

    printf("%8s %12s %12s %12s %12s %12s %12s %12s %12s\n",
           "N", "elems", "pi_wall(s)", "blas_wall", "pi_cpu(s)", "blas_cpu", "pi_GF/s", "blas_GF/s", "max_err");
    for (size_t i = 0; i < n_scales; i++)
    {
        printf("%8zu %12zu %12.6f %12.6f %12.6f %12.6f %12.3f %12.3f %12.3e\n",
               rows[i].N, rows[i].elems, rows[i].pi_wall, rows[i].blas_wall,
               rows[i].pi_cpu, rows[i].blas_cpu, rows[i].pi_gflops, rows[i].blas_gflops, rows[i].max_err);
    }

    return 0;
}