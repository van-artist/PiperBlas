#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include "pi_blas.h"
#include "pi_csr.h"
#include "pi_type.h"

static double wall_now_gettimeofday(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

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

typedef struct
{
    double avg, std, best;
} Stats;

static Stats summarize(const std::vector<double> &v)
{
    if (v.empty())
        return {0, 0, 0};
    double sum = 0.0, sum2 = 0.0, best = v[0];
    for (double x : v)
    {
        sum += x;
        sum2 += x * x;
        if (x < best)
            best = x;
    }
    double n = (double)v.size();
    double avg = sum / n;
    double var = fmax(0.0, sum2 / n - avg * avg);
    return {avg, sqrt(var), best};
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "用法: %s <matrix.bin> [warmup=2] [iters=10] [cycles=3]\n", argv[0]);
        return 1;
    }

    const char *bin_path = argv[1];
    int warmup = (argc >= 3) ? atoi(argv[2]) : 2;
    int iters = (argc >= 4) ? atoi(argv[3]) : 10;
    int cycles = (argc >= 5) ? atoi(argv[4]) : 3;

    pi_csr A;
    piState st = csr_from_bin(bin_path, &A);
    if (st != piSuccess)
    {
        fprintf(stderr, "csr_from_bin 读取失败: %s\n", bin_path);
        return 1;
    }
    size_t m = A.n_rows;
    size_t n = A.n_cols;
    size_t nnz = A.nnz;
    printf("已加载矩阵: %zu x %zu, nnz = %zu\n", m, n, nnz);

    double *x = (double *)xalloc(n * sizeof(double));
    double *y1 = (double *)xalloc(m * sizeof(double)); // piSpMV 输出
    double *y2 = (double *)xalloc(m * sizeof(double)); // MKL SpMV 输出

    srand(1);
    for (size_t i = 0; i < n; ++i)
        x[i] = ((double)(rand()) / RAND_MAX) * 2.0 - 1.0;

    // ======== 构造 MKL CSR 矩阵 ========
    MKL_INT m_mkl = (MKL_INT)m;
    MKL_INT n_mkl = (MKL_INT)n;

    std::vector<MKL_INT> mkl_row_ptr(m + 1);
    std::vector<MKL_INT> mkl_col_idx(nnz);
    std::vector<double> mkl_values(nnz);

    for (size_t i = 0; i <= m; ++i)
        mkl_row_ptr[i] = (MKL_INT)A.row_ptr[i];
    for (size_t j = 0; j < nnz; ++j)
    {
        mkl_col_idx[j] = (MKL_INT)A.col_idx[j];
        mkl_values[j] = A.values[j];
    }

    sparse_matrix_t Amkl;
    sparse_status_t status = mkl_sparse_d_create_csr(
        &Amkl,
        SPARSE_INDEX_BASE_ZERO,
        m_mkl,
        n_mkl,
        mkl_row_ptr.data(),
        mkl_row_ptr.data() + 1,
        mkl_col_idx.data(),
        mkl_values.data());

    if (status != SPARSE_STATUS_SUCCESS)
    {
        fprintf(stderr, "mkl_sparse_d_create_csr 失败, status = %d\n", status);
        csr_destroy(&A);
        free(x);
        free(y1);
        free(y2);
        return 1;
    }

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    mkl_sparse_optimize(Amkl);

    // ======== 先做一次正确性验证 ========
    memset(y1, 0, m * sizeof(double));
    memset(y2, 0, m * sizeof(double));

    piSpMV(&A, x, y1);

    status = mkl_sparse_d_mv(
        SPARSE_OPERATION_NON_TRANSPOSE,
        1.0,
        Amkl,
        descr,
        x,
        0.0,
        y2);

    if (status != SPARSE_STATUS_SUCCESS)
    {
        fprintf(stderr, "mkl_sparse_d_mv 失败, status = %d\n", status);
        mkl_sparse_destroy(Amkl);
        csr_destroy(&A);
        free(x);
        free(y1);
        free(y2);
        return 1;
    }

    double max_err = 0.0, l2 = 0.0;
    for (size_t i = 0; i < m; ++i)
    {
        double d = fabs(y1[i] - y2[i]);
        if (d > max_err)
            max_err = d;
        l2 += d * d;
    }
    l2 = sqrt(l2);
    printf("结果验证: max_err = %.3e, l2 = %.3e\n", max_err, l2);

    const double ops = 2.0 * (double)nnz;

    // ======== warmup ========
    for (int k = 0; k < warmup; ++k)
    {
        piSpMV(&A, x, y1);
        mkl_sparse_d_mv(
            SPARSE_OPERATION_NON_TRANSPOSE,
            1.0,
            Amkl,
            descr,
            x,
            0.0,
            y2);
    }

    std::vector<double> pi_cycle_avg, mkl_cycle_avg;
    std::vector<double> pi_all, mkl_all;

    // ======== 正式计时 ========
    for (int c = 0; c < cycles; ++c)
    {
        double sum_pi = 0.0;
        double sum_mkl = 0.0;

        for (int it = 0; it < iters; ++it)
        {
            double t0 = wall_now_gettimeofday();
            piSpMV(&A, x, y1);
            double t1 = wall_now_gettimeofday();
            sum_pi += (t1 - t0);
            pi_all.push_back(t1 - t0);

            double t2 = wall_now_gettimeofday();
            mkl_sparse_d_mv(
                SPARSE_OPERATION_NON_TRANSPOSE,
                1.0,
                Amkl,
                descr,
                x,
                0.0,
                y2);
            double t3 = wall_now_gettimeofday();
            sum_mkl += (t3 - t2);
            mkl_all.push_back(t3 - t2);
        }

        pi_cycle_avg.push_back(sum_pi / iters);
        mkl_cycle_avg.push_back(sum_mkl / iters);
    }

    Stats s_pi = summarize(pi_cycle_avg);
    Stats s_mkl = summarize(mkl_cycle_avg);
    Stats s_pi_all = summarize(pi_all);
    Stats s_mkl_all = summarize(mkl_all);

    double gflops_pi_avg = (ops / s_pi.avg) * 1e-9;
    double gflops_mkl_avg = (ops / s_mkl.avg) * 1e-9;
    double gflops_pi_best = (ops / s_pi_all.best) * 1e-9;
    double gflops_mkl_best = (ops / s_mkl_all.best) * 1e-9;

    printf("测试配置: warmup=%d iters/周期=%d cycles=%d\n", warmup, iters, cycles);
    printf("PI:   平均=%.6f s  Std=%.6f  Best=%.6f s  ⇒  Avg=%.3f GF/s  Best=%.3f GF/s\n",
           s_pi.avg, s_pi.std, s_pi_all.best, gflops_pi_avg, gflops_pi_best);
    printf("MKL:  平均=%.6f s  Std=%.6f  Best=%.6f s  ⇒  Avg=%.3f GF/s  Best=%.3f GF/s\n",
           s_mkl.avg, s_mkl.std, s_mkl_all.best, gflops_mkl_avg, gflops_mkl_best);

    mkl_sparse_destroy(Amkl);
    free(x);
    free(y1);
    free(y2);
    csr_destroy(&A);
    return 0;
}
