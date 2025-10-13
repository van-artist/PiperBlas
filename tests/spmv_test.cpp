#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "utils.hpp"
#include "pi_blas.hpp"

static double wall_now()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}
static double cpu_now() { return (double)clock() / (double)CLOCKS_PER_SEC; }
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
static void fill_rand(double *x, size_t n, uint64_t *seed)
{
    for (size_t i = 0; i < n; i++)
    {
        *seed = (*seed * 2862933555777941757ULL) + 3037000493ULL;
        x[i] = ((double)(*seed >> 11) / (double)(1ULL << 53)) * 2.0 - 1.0;
    }
}
static void spmv_ref(const size_t *col, const size_t *rowptr, const double *val, const double *x, double *y, size_t rows)
{
    for (size_t i = 0; i < rows; i++)
    {
        double s = 0.0;
        for (size_t p = rowptr[i]; p < rowptr[i + 1]; p++)
            s += val[p] * x[col[p]];
        y[i] = s;
    }
}
static int cmp_size_t(const void *a, const void *b)
{
    size_t x = *(const size_t *)a, y = *(const size_t *)b;
    return (x > y) - (x < y);
}

int main()
{
    const size_t Ns[] = {1024, 4096, 16384, 65536};
    const size_t Krow[] = {8, 16, 16, 32};
    const size_t T = sizeof(Ns) / sizeof(Ns[0]);
    const size_t DENSE_BYTES_LIMIT = 512ULL * 1024ULL * 1024ULL;

    printf("%8s %8s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n",
           "rows", "krow", "nnz", "pi_wall", "pi_cpu", "ref_wall", "ref_cpu", "sp_GF/s", "blas_wall", "blas_cpu", "blas_GF/s", "max_err");

    for (size_t t = 0; t < T; t++)
    {
        size_t rows = Ns[t], cols = Ns[t], krow = Krow[t];
        size_t nnz = rows * krow;

        size_t *rowptr = (size_t *)xalloc((rows + 1) * sizeof(size_t));
        size_t *colidx = (size_t *)xalloc(nnz * sizeof(size_t));
        double *aval = (double *)xalloc(nnz * sizeof(double));
        double *x = (double *)xalloc(cols * sizeof(double));
        double *y_pi = (double *)xalloc(rows * sizeof(double));
        double *y_ref = (double *)xalloc(rows * sizeof(double));
        double *y_blas = (double *)xalloc(rows * sizeof(double));

        uint64_t seed = 1;
        fill_rand(x, cols, &seed);

        rowptr[0] = 0;
        for (size_t i = 0; i < rows; i++)
        {
            rowptr[i + 1] = rowptr[i] + krow;
            size_t base = rowptr[i];
            for (size_t j = 0; j < krow; j++)
            {
                seed = (seed * 2862933555777941757ULL) + 3037000493ULL;
                colidx[base + j] = (size_t)((seed >> 11) % cols);
            }
            qsort(colidx + base, krow, sizeof(size_t), cmp_size_t);
            size_t w = base;
            for (size_t j = base; j < base + krow; j++)
            {
                if (j == base || colidx[j] != colidx[j - 1])
                    colidx[w++] = colidx[j];
            }
            while (w < base + krow)
            {
                size_t tmp = (colidx[base] + (w - base)) % cols;
                colidx[w++] = tmp;
            }
            for (size_t j = base; j < base + krow; j++)
            {
                seed = (seed * 2862933555777941757ULL) + 3037000493ULL;
                aval[j] = ((double)(seed >> 11) / (double)(1ULL << 53)) * 2.0 - 1.0;
            }
        }

        double w0 = wall_now(), c0 = cpu_now();
        piSpMV(colidx, rowptr, aval, x, y_pi, nnz, rows + 1, cols, rows);
        double c1 = cpu_now(), w1 = wall_now();

        double w2 = wall_now(), c2 = cpu_now();
        spmv_ref(colidx, rowptr, aval, x, y_ref, rows);
        double c3 = cpu_now(), w3 = wall_now();

        double blas_wall = NAN, blas_cpu = NAN, blas_gfs = NAN;
        size_t dense_elems = rows * cols;
        if ((double)dense_elems <= (double)(DENSE_BYTES_LIMIT / sizeof(double)))
        {
            double *A = (double *)xalloc(dense_elems * sizeof(double));
            memset(A, 0, dense_elems * sizeof(double));
            for (size_t i = 0; i < rows; i++)
            {
                for (size_t p = rowptr[i]; p < rowptr[i + 1]; p++)
                {
                    A[i * cols + colidx[p]] = aval[p];
                }
            }
            double wb0 = wall_now(), cb0 = cpu_now();
            cblas_dgemv(CblasRowMajor, CblasNoTrans, (int)rows, (int)cols, 1.0, A, (int)cols, x, 1, 0.0, y_blas, 1);
            double cb1 = cpu_now(), wb1 = wall_now();
            blas_wall = wb1 - wb0;
            blas_cpu = cb1 - cb0;
            double dense_ops = 2.0 * (double)rows * (double)cols;
            blas_gfs = dense_ops / blas_wall * 1e-9;
            free(A);
        }

        double max_err = 0.0;
        for (size_t i = 0; i < rows; i++)
        {
            double d = fabs(y_pi[i] - y_ref[i]);
            if (d > max_err)
                max_err = d;
        }
        double sp_ops = 2.0 * (double)nnz;
        double sp_gfs = sp_ops / (w1 - w0) * 1e-9;

        printf("%8zu %8zu %12zu %12.6f %12.6f %12.6f %12.6f %12.3f %12.6f %12.6f %12.3f %12.3e\n",
               rows, krow, nnz, (w1 - w0), (c1 - c0), (w3 - w2), (c3 - c2), sp_gfs,
               (isnan(blas_wall) ? 0.0 : blas_wall), (isnan(blas_cpu) ? 0.0 : blas_cpu), (isnan(blas_gfs) ? 0.0 : blas_gfs), max_err);

        free(rowptr);
        free(colidx);
        free(aval);
        free(x);
        free(y_pi);
        free(y_ref);
        free(y_blas);
    }
    return 0;
}