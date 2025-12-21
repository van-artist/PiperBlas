#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <ctime>

#include "pi_blas.hpp"
#include "pi_csr.hpp"
#include "pi_type.hpp"
#include "core/pi_config.hpp"
#include "core/test_utils.hpp"

#if defined(PIPER_HAVE_EIGEN) && PIPER_HAVE_EIGEN

#include <Eigen/Sparse>

struct Result
{
    std::string name;
    int m, n, nnz;
    double gflops_pi;
    double gflops_eigen;
};

static Result run_case(const char *bin_path, int warmup, int iters)
{
    pi_csr A;
    if (csr_from_bin(bin_path, &A) != piSuccess)
    {
        std::fprintf(stderr, "failed to load %s\n", bin_path);
        std::exit(1);
    }

    int m = A.n_rows;
    int n = A.n_cols;
    int nnz = A.nnz;

    double *x = (double *)aligned_alloc64((size_t)n * sizeof(double));
    double *y_pi = (double *)aligned_alloc64((size_t)m * sizeof(double));
    double *y_eigen = (double *)aligned_alloc64((size_t)m * sizeof(double));

    uint64_t seed = 1;
    fill_random_double(x, (size_t)n, &seed);

    using MapSparse = Eigen::MappedSparseMatrix<double, Eigen::RowMajor, int>;
    MapSparse Aeigen(m, n, nnz, A.row_ptr, A.col_idx, A.values);
    Eigen::Map<const Eigen::VectorXd> x_map(x, n);
    Eigen::Map<Eigen::VectorXd> y_eigen_map(y_eigen, m);

    for (int i = 0; i < warmup; ++i)
    {
        piSpMV(&A, x, y_pi);
        y_eigen_map.noalias() = Aeigen * x_map;
    }

    auto time_cpu_s = [&](auto fn)
    {
        double t0 = (double)clock() / (double)CLOCKS_PER_SEC;
        for (int i = 0; i < iters; ++i)
            fn();
        double t1 = (double)clock() / (double)CLOCKS_PER_SEC;
        return (t1 - t0) / (double)iters;
    };

    double pi_s = time_cpu_s([&]() { piSpMV(&A, x, y_pi); });
    double eigen_s = time_cpu_s([&]() { y_eigen_map.noalias() = Aeigen * x_map; });

    Result out{};
    out.name = bin_path;
    out.m = m;
    out.n = n;
    out.nnz = nnz;
    const double ops = 2.0 * (double)nnz;
    out.gflops_pi = ops / pi_s * 1e-9;
    out.gflops_eigen = ops / eigen_s * 1e-9;

    free(x);
    free(y_pi);
    free(y_eigen);
    csr_destroy(&A);
    return out;
}

int main(int argc, char **argv)
{
    config_init();

    if (argc < 2)
    {
        std::fprintf(stderr, "用法: %s <matrix.bin|dir> [--warmup=2] [--iters=10]\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    int warmup = parse_int_flag(argc, argv, "--warmup=", 2);
    int iters = parse_int_flag(argc, argv, "--iters=", 10);

    std::vector<std::string> inputs;
    if (is_directory(path))
        inputs = list_files_in_dir(path);
    else
        inputs.push_back(path);

    if (inputs.empty())
    {
        std::fprintf(stderr, "没有找到输入文件: %s\n", path);
        return 1;
    }

    std::vector<Result> results;
    results.reserve(inputs.size());
    for (auto &p : inputs)
        results.push_back(run_case(p.c_str(), warmup, iters));

    std::printf("==== SpMV CPU (Eigen vs pi) ====\n");
    std::printf("%32s | %8s %8s %10s | %10s %10s\n", "file", "m", "n", "nnz", "pi_GF/s", "eigen_GF/s");

    for (const auto &r : results)
    {
        const char *name = r.name.c_str();
        const char *base = std::strrchr(name, '/');
        base = base ? base + 1 : name;

        std::printf("%32s | %8d %8d %10d | %10.3f %10.3f\n",
                    base, r.m, r.n, r.nnz,
                    r.gflops_pi, r.gflops_eigen);
    }

    return 0;
}

#elif defined(PIPER_HAVE_MKL) && PIPER_HAVE_MKL

#include <mkl.h>

struct Result
{
    std::string name;
    int m, n, nnz;
    double gflops_pi;
    double gflops_mkl;
};

static Result run_case(const char *bin_path, int warmup, int iters)
{
    pi_csr A;
    if (csr_from_bin(bin_path, &A) != piSuccess)
    {
        std::fprintf(stderr, "failed to load %s\n", bin_path);
        std::exit(1);
    }

    int m = A.n_rows;
    int n = A.n_cols;
    int nnz = A.nnz;

    double *x = (double *)aligned_alloc64((size_t)n * sizeof(double));
    double *y_pi = (double *)aligned_alloc64((size_t)m * sizeof(double));
    double *y_mkl = (double *)aligned_alloc64((size_t)m * sizeof(double));

    uint64_t seed = 1;
    fill_random_double(x, (size_t)n, &seed);

    sparse_matrix_t Amkl;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_d_create_csr(&Amkl, SPARSE_INDEX_BASE_ZERO,
                            m, n,
                            (int *)A.row_ptr,
                            (int *)A.row_ptr + 1,
                            (int *)A.col_idx,
                            (double *)A.values);

    for (int i = 0; i < warmup; ++i)
    {
        piSpMV(&A, x, y_pi);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, Amkl, descr, x, 0.0, y_mkl);
    }

    auto time_cpu_s = [&](auto fn)
    {
        double t0 = (double)clock() / (double)CLOCKS_PER_SEC;
        for (int i = 0; i < iters; ++i)
            fn();
        double t1 = (double)clock() / (double)CLOCKS_PER_SEC;
        return (t1 - t0) / (double)iters;
    };

    double pi_s = time_cpu_s([&]() { piSpMV(&A, x, y_pi); });
    double mkl_s = time_cpu_s([&]() { mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, Amkl, descr, x, 0.0, y_mkl); });

    Result out{};
    out.name = bin_path;
    out.m = m;
    out.n = n;
    out.nnz = nnz;
    const double ops = 2.0 * (double)nnz;
    out.gflops_pi = ops / pi_s * 1e-9;
    out.gflops_mkl = ops / mkl_s * 1e-9;

    mkl_sparse_destroy(Amkl);
    free(x);
    free(y_pi);
    free(y_mkl);
    csr_destroy(&A);
    return out;
}

int main(int argc, char **argv)
{
    config_init();

    if (argc < 2)
    {
        std::fprintf(stderr, "用法: %s <matrix.bin|dir> [--warmup=2] [--iters=10]\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    int warmup = parse_int_flag(argc, argv, "--warmup=", 2);
    int iters = parse_int_flag(argc, argv, "--iters=", 10);

    std::vector<std::string> inputs;
    if (is_directory(path))
        inputs = list_files_in_dir(path);
    else
        inputs.push_back(path);

    if (inputs.empty())
    {
        std::fprintf(stderr, "没有找到输入文件: %s\n", path);
        return 1;
    }

    std::vector<Result> results;
    results.reserve(inputs.size());
    for (auto &p : inputs)
        results.push_back(run_case(p.c_str(), warmup, iters));

    std::printf("==== SpMV CPU (MKL vs pi) ====\n");
    std::printf("%32s | %8s %8s %10s | %10s %10s\n", "file", "m", "n", "nnz", "pi_GF/s", "mkl_GF/s");

    for (const auto &r : results)
    {
        const char *name = r.name.c_str();
        const char *base = std::strrchr(name, '/');
        base = base ? base + 1 : name;

        std::printf("%32s | %8d %8d %10d | %10.3f %10.3f\n",
                    base, r.m, r.n, r.nnz,
                    r.gflops_pi, r.gflops_mkl);
    }

    return 0;
}

#else

int main()
{
    std::fprintf(stderr, "Neither Eigen3 nor MKL available; SpMV test disabled.\n");
    return 0;
}

#endif
