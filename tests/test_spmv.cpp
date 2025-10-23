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
#include "pi_csr.h"
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
    if (argc < 2)
    {
        fprintf(stderr, "用法: %s <matrix.bin> [density]\n", argv[0]);
        return 1;
    }

    const char *bin_path = argv[1]; // <-- 新增：从命令行读取 bin 文件路径
    double density = 0.01;
    if (argc >= 3)
        density = atof(argv[2]);

    // ================================
    // 载入 CSR 矩阵
    // ================================
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
    double *y1 = (double *)xalloc(m * sizeof(double));
    double *y2 = (double *)xalloc(m * sizeof(double));

    // 填充 x 为随机数
    uint64_t seed = 1;
    for (size_t i = 0; i < n; ++i)
        x[i] = ((double)(rand()) / RAND_MAX) * 2.0 - 1.0;

    memset(y1, 0, m * sizeof(double));
    memset(y2, 0, m * sizeof(double));

    // Eigen 对照验证
    typedef Eigen::Triplet<double, int> T;
    std::vector<T> trips;
    trips.reserve(nnz);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j)
            trips.emplace_back((int)i, (int)A.col_idx[j], A.values[j]);

    Eigen::SparseMatrix<double, Eigen::RowMajor, int> Ae((int)m, (int)n);
    Ae.setFromTriplets(trips.begin(), trips.end());

    // ================================
    // 自家实现：piSpMV
    // ================================
    double w0 = wall_now_gettimeofday();
    double c0 = cpu_now_clock();
    piSpMV(&A, x, y1);
    double c1 = cpu_now_clock();
    double w1 = wall_now_gettimeofday();

    // ================================
    // Eigen 实现
    // ================================
    Eigen::Map<const Eigen::VectorXd> X(x, (Eigen::Index)n);
    Eigen::Map<Eigen::VectorXd> Y2(y2, (Eigen::Index)m);
    double w2 = wall_now_gettimeofday();
    double c2 = cpu_now_clock();
    Y2 = Ae * X;
    double c3 = cpu_now_clock();
    double w3 = wall_now_gettimeofday();

    // ================================
    // 误差对比与性能
    // ================================
    double max_err = 0.0, l2 = 0.0;
    for (size_t i = 0; i < m; ++i)
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

    printf("结果验证: max_err = %.3e, l2 = %.3e\n", max_err, l2);
    printf("性能: pi_wall=%.6f eig_wall=%.6f pi_GF/s=%.3f eig_GF/s=%.3f\n",
           pi_wall, eig_wall, gflops_pi, gflops_eig);

    // ================================
    // 资源释放
    // ================================
    free(x);
    free(y1);
    free(y2);
    csr_destroy(&A);

    return 0;
}