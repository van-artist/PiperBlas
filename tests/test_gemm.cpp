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

// 结果记录结构
typedef struct
{
    size_t N, elems;
    double wall[3], cpu[3], gflops[3]; // v1,v2,v3
    double blas_wall, blas_cpu, blas_gflops;
    double max_err[3], l2[3];
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
    const size_t Ns[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    const size_t n_scales = sizeof(Ns) / sizeof(Ns[0]);
    const double alpha = 1.2, beta = 0.8;
    Row rows[n_scales];

    for (size_t t = 0; t < n_scales; t++)
    {
        size_t N = Ns[t];
        size_t m = N, k = N, n = N;
        size_t asz = m * k, bsz = k * n, csz = m * n;
        double *A = (double *)xalloc(asz * sizeof(double));
        double *B = (double *)xalloc(bsz * sizeof(double));
        double *Cref = (double *)xalloc(csz * sizeof(double));
        double *Ctmp = (double *)xalloc(csz * sizeof(double));

        uint64_t seed = 1;
        fill(A, asz, &seed);
        fill(B, bsz, &seed);
        fill(Cref, csz, &seed);

        // 基准 BLAS
        for (size_t i = 0; i < csz; i++)
            Ctmp[i] = Cref[i];
        double w0 = wall_now_gettimeofday(), c0 = cpu_now_clock();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k,
                    alpha, A, (int)k, B, (int)n, beta, Ctmp, (int)n);
        double c1 = cpu_now_clock(), w1 = wall_now_gettimeofday();
        double blas_wall = w1 - w0, blas_cpu = c1 - c0;
        double ops = 2.0 * (double)m * (double)k * (double)n;
        double blas_gflops = ops / blas_wall * 1e-9;

        // 三个版本依次测试
        piState (*funcs[3])(double *, double *, double *, double, double, size_t, size_t, size_t) =
            {piGemm, piGemm_v2, piGemm_v3};
        for (int v = 0; v < 3; v++)
        {
            double *C = (double *)xalloc(csz * sizeof(double));
            for (size_t i = 0; i < csz; i++)
                C[i] = Cref[i];

            double w2 = wall_now_gettimeofday(), c2 = cpu_now_clock();
            funcs[v](A, B, C, alpha, beta, m, k, n);
            double c3 = cpu_now_clock(), w3 = wall_now_gettimeofday();

            rows[t].wall[v] = w3 - w2;
            rows[t].cpu[v] = c3 - c2;
            rows[t].gflops[v] = ops / rows[t].wall[v] * 1e-9;

            double max_err = 0, l2 = 0;
            for (size_t i = 0; i < csz; i++)
            {
                double d = fabs(C[i] - Ctmp[i]);
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

        free(A);
        free(B);
        free(Cref);
        free(Ctmp);
    }

    printf("%8s %8s | %10s %10s %10s | %10s %10s %10s | %10s\n",
           "N", "elems", "v1_GF/s", "v2_GF/s", "v3_GF/s", "blas_GF/s", "v1_err", "v2_err", "v3_err");

    for (size_t i = 0; i < n_scales; i++)
    {
        printf("%8zu %8zu | %10.3f %10.3f %10.3f | %10.3f %10.3e %10.3e %10.3e\n",
               rows[i].N, rows[i].elems,
               rows[i].gflops[0], rows[i].gflops[1], rows[i].gflops[2],
               rows[i].blas_gflops,
               rows[i].max_err[0], rows[i].max_err[1], rows[i].max_err[2]);
    }

    return 0;
}
