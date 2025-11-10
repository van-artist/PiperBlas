#include <Eigen/Sparse>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <vector>
#include <algorithm>

extern "C"
{
#include "pi_blas.h"
#include "pi_csr.h"
#include "pi_type.h"
}

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
    size_t m = A.n_rows, n = A.n_cols, nnz = A.nnz;
    printf("已加载矩阵: %zu x %zu, nnz = %zu\n", m, n, nnz);

    double *x = (double *)xalloc(n * sizeof(double));
    double *y1 = (double *)xalloc(m * sizeof(double)); // 用于两个实现输出
    double *y2 = (double *)xalloc(m * sizeof(double)); // Eigen 输出/真值
    double *yref = y2;                                 // 直接把 y2 作为参考向量

    srand(1);
    for (size_t i = 0; i < n; ++i)
        x[i] = ((double)(rand()) / RAND_MAX) * 2.0 - 1.0;

    // 构建 Eigen 稀疏矩阵
    typedef Eigen::Triplet<double, int> T;
    std::vector<T> trips;
    trips.reserve(nnz);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j)
            trips.emplace_back((int)i, (int)A.col_idx[j], A.values[j]);
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> Ae((int)m, (int)n);
    Ae.setFromTriplets(trips.begin(), trips.end());
    Eigen::Map<const Eigen::VectorXd> X(x, (Eigen::Index)n);
    Eigen::Map<Eigen::VectorXd> Yref(yref, (Eigen::Index)m);

    // 生成参考结果（不计时）
    memset(yref, 0, m * sizeof(double));
    Yref = Ae * X;

    // 计算量（SpMV ~ 2*nnz FLOPs）
    const double ops = 2.0 * (double)nnz;

    struct Impl
    {
        const char *name;
        piState (*fn)(const pi_csr *, double *, double *);
    } impls[] = {
        {"PI pthread", &piSpMV},   // 你的 pthread 版本
        {"PI OpenMP", &piSpMV_v2}, // 你的 OpenMP 版本
    };
    const int NIMPL = (int)(sizeof(impls) / sizeof(impls[0]));

    // 先对每个实现做一次正确性验证
    for (int id = 0; id < NIMPL; ++id)
    {
        memset(y1, 0, m * sizeof(double));
        impls[id].fn(&A, x, y1);
        double max_err = 0.0, l2 = 0.0;
        for (size_t i = 0; i < m; ++i)
        {
            double d = fabs(y1[i] - yref[i]);
            if (d > max_err)
                max_err = d;
            l2 += d * d;
        }
        l2 = sqrt(l2);
        printf("[%s] 结果验证: max_err = %.3e, l2 = %.3e\n", impls[id].name, max_err, l2);
    }

    // Eigen 版本的计时统计
    std::vector<double> eig_cycle_avg, eig_all;
    for (int k = 0; k < warmup; ++k)
    { // warmup
        Yref = Ae * X;
    }
    for (int c = 0; c < cycles; ++c)
    {
        double sum = 0.0;
        for (int it = 0; it < iters; ++it)
        {
            double t0 = wall_now_gettimeofday();
            Yref = Ae * X;
            double t1 = wall_now_gettimeofday();
            sum += (t1 - t0);
            eig_all.push_back(t1 - t0);
        }
        eig_cycle_avg.push_back(sum / iters);
    }
    Stats s_eig = summarize(eig_cycle_avg);
    Stats s_eig_all = summarize(eig_all);
    double gflops_eig_avg = (ops / s_eig.avg) * 1e-9;
    double gflops_eig_best = (ops / s_eig_all.best) * 1e-9;

    // 两个 PI 实现的计时统计
    for (int id = 0; id < NIMPL; ++id)
    {
        std::vector<double> cyc_avg, all;
        // warmup
        for (int k = 0; k < warmup; ++k)
        {
            impls[id].fn(&A, x, y1);
        }
        // 正式计时
        for (int c = 0; c < cycles; ++c)
        {
            double sum = 0.0;
            for (int it = 0; it < iters; ++it)
            {
                double t0 = wall_now_gettimeofday();
                impls[id].fn(&A, x, y1);
                double t1 = wall_now_gettimeofday();
                sum += (t1 - t0);
                all.push_back(t1 - t0);
            }
            cyc_avg.push_back(sum / iters);
        }
        Stats s = summarize(cyc_avg);
        Stats s_all = summarize(all);
        double gflops_avg = (ops / s.avg) * 1e-9;
        double gflops_best = (ops / s_all.best) * 1e-9;

        printf("%s: 平均=%.6f s  Std=%.6f  Best=%.6f s  ⇒  Avg=%.3f GF/s  Best=%.3f GF/s\n",
               impls[id].name, s.avg, s.std, s_all.best, gflops_avg, gflops_best);
    }

    // Eigen 结果
    printf("Eigen:平均=%.6f s  Std=%.6f  Best=%.6f s  ⇒  Avg=%.3f GF/s  Best=%.3f GF/s\n",
           s_eig.avg, s_eig.std, s_eig_all.best, gflops_eig_avg, gflops_eig_best);

    free(x);
    free(y1);
    free(y2);
    csr_destroy(&A);
    return 0;
}
