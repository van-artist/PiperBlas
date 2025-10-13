#include <Eigen/Sparse>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
extern "C"
{
#include "pi_blas.h"
#include "pi_type.h"
}

static double wall_now_gettimeofday(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static double cpu_now_clock(void) { return (double)clock() / (double)CLOCKS_PER_SEC; }

static void *xalloc(size_t nbytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, nbytes) != 0 || !p)
    {
        fprintf(stderr, "alloc fail\n");
        exit(1);
    }
    return p;
}

static inline double rnd(uint64_t *seed)
{
    *seed = (*seed * 2862933555777941757ULL) + 3037000493ULL;
    return ((double)(*seed >> 11) / (double)(1ULL << 53)) * 2.0 - 1.0;
}

static void fill(double *x, size_t n, uint64_t *seed)
{
    for (size_t i = 0; i < n; i++)
        x[i] = rnd(seed);
}

static CSR *csr_random(size_t n_rows, size_t n_cols, double density, uint64_t *seed)
{
    if (density <= 0.0)
        density = 1.0 / (double)n_cols;
    if (density > 1.0)
        density = 1.0;
    size_t *row_ptr = (size_t *)xalloc((n_rows + 1) * sizeof(size_t));
    size_t nnz = 0;
    for (size_t i = 0; i < n_rows; ++i)
    {
        size_t cnt = 0;
        for (size_t j = 0; j < n_cols; ++j)
        {
            double u = rnd(seed) * 0.5 + 0.5;
            if (u < density)
                ++cnt;
        }
        if (cnt == 0)
            cnt = 1;
        row_ptr[i] = nnz;
        nnz += cnt;
    }
    row_ptr[n_rows] = nnz;
    size_t *col_idx = (size_t *)xalloc(nnz * sizeof(size_t));
    double *values = (double *)xalloc(nnz * sizeof(double));
    size_t pos = 0;
    for (size_t i = 0; i < n_rows; ++i)
    {
        size_t start = pos;
        size_t target = row_ptr[i + 1] - row_ptr[i];
        size_t trials = 0;
        while (pos - start < target && trials < target * 20 + 100)
        {
            size_t c = (size_t)((rnd(seed) * 0.5 + 0.5) * (double)n_cols);
            if (c >= n_cols)
                c = n_cols - 1;
            int dup = 0;
            for (size_t t = start; t < pos; ++t)
            {
                if (col_idx[t] == c)
                {
                    dup = 1;
                    break;
                }
            }
            if (!dup)
                col_idx[pos++] = c;
            ++trials;
        }
        for (; pos - start < target; ++pos)
        {
            col_idx[pos] = (size_t)(((pos - start) * (n_cols / (target ? target : 1))) % n_cols);
            if (col_idx[pos] >= n_cols)
                col_idx[pos] = n_cols - 1;
            for (size_t t = start; t < pos; ++t)
            {
                if (col_idx[t] == col_idx[pos])
                {
                    col_idx[pos] = (col_idx[pos] + 1) % n_cols;
                    t = start - 1;
                }
            }
        }
        for (size_t a = start + 1; a < pos; ++a)
        {
            size_t key = col_idx[a];
            size_t b = a;
            while (b > start && col_idx[b - 1] > key)
            {
                col_idx[b] = col_idx[b - 1];
                --b;
            }
            col_idx[b] = key;
        }
        for (size_t t = start; t < pos; ++t)
            values[t] = rnd(seed);
    }
    CSR *A = (CSR *)xalloc(sizeof(CSR));
    A->n_rows = n_rows;
    A->n_cols = n_cols;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_idx = col_idx;
    A->values = values;
    return A;
}

static void csr_destroy(CSR *A)
{
    if (!A)
        return;
    free(A->row_ptr);
    free(A->col_idx);
    free(A->values);
    free(A);
}

typedef struct
{
    size_t N, m, n, nnz;
    double density;
    double pi_wall, eig_wall;
    double pi_cpu, eig_cpu;
    double pi_gflops, eig_gflops;
    double max_err, l2;
} Row;

int main(int argc, char **argv)
{
    const size_t Ns[] = {512, 1024, 2048, 4096};
    const size_t n_scales = sizeof(Ns) / sizeof(Ns[0]);
    double density = 0.01;
    if (argc >= 2)
        density = atof(argv[1]);
    Row rows[n_scales];
    for (size_t t = 0; t < n_scales; t++)
    {
        size_t N = Ns[t];
        size_t m = N, n = N;
        uint64_t seed = 1;
        CSR *A = csr_random(m, n, density, &seed);
        size_t nnz = A->nnz;
        double *x = (double *)xalloc(n * sizeof(double));
        double *y1 = (double *)xalloc(m * sizeof(double));
        double *y2 = (double *)xalloc(m * sizeof(double));
        fill(x, n, &seed);
        for (size_t i = 0; i < m; ++i)
        {
            y1[i] = 0.0;
            y2[i] = 0.0;
        }
        typedef Eigen::Triplet<double, int> T;
        std::vector<T> trips;
        trips.reserve((size_t)(nnz));
        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = A->row_ptr[i]; j < A->row_ptr[i + 1]; ++j)
            {
                trips.emplace_back((int)i, (int)A->col_idx[j], A->values[j]);
            }
        }
        Eigen::SparseMatrix<double, Eigen::RowMajor, int> Ae((int)m, (int)n);
        Ae.setFromTriplets(trips.begin(), trips.end());
        double w0 = wall_now_gettimeofday();
        double c0 = cpu_now_clock();
        piSpMV(A, x, y1, n, m);
        double c1 = cpu_now_clock();
        double w1 = wall_now_gettimeofday();
        Eigen::Map<const Eigen::VectorXd> X(x, (Eigen::Index)n);
        Eigen::Map<Eigen::VectorXd> Y2(y2, (Eigen::Index)m);
        double w2 = wall_now_gettimeofday();
        double c2 = cpu_now_clock();
        Y2 = Ae * X;
        double c3 = cpu_now_clock();
        double w3 = wall_now_gettimeofday();
        double max_err = 0.0, l2 = 0.0;
        for (size_t i = 0; i < m; i++)
        {
            double d = fabs(y1[i] - y2[i]);
            if (d > max_err)
                max_err = d;
            l2 += d * d;
        }
        l2 = sqrt(l2);
        double ops = 2.0 * (double)nnz;
        double pi_wall = w1 - w0, eig_wall = w3 - w2;
        double pi_cpu = c1 - c0, eig_cpu = c3 - c2;
        double gflops_pi = (ops / pi_wall) * 1e-9;
        double gflops_eig = (ops / eig_wall) * 1e-9;
        rows[t] = (Row){N, m, n, nnz, density, pi_wall, eig_wall, pi_cpu, eig_cpu, gflops_pi, gflops_eig, max_err, l2};
        csr_destroy(A);
        free(x);
        free(y1);
        free(y2);
    }
    printf("%6s %8s %8s %10s %12s %12s %12s %12s %12s %12s %12s\n",
           "N", "m", "n", "nnz", "density",
           "pi_wall", "eig_wall", "pi_cpu", "eig_cpu", "pi_GF/s", "eig_GF/s");
    for (size_t i = 0; i < n_scales; i++)
    {
        printf("%6zu %8zu %8zu %10zu %12.4f %12.6f %12.6f %12.6f %12.6f %12.3f %12.3f   max_err=%8.2e  l2=%8.2e\n",
               rows[i].N, rows[i].m, rows[i].n, rows[i].nnz, rows[i].density,
               rows[i].pi_wall, rows[i].eig_wall, rows[i].pi_cpu, rows[i].eig_cpu,
               rows[i].pi_gflops, rows[i].eig_gflops, rows[i].max_err, rows[i].l2);
    }
    return 0;
}