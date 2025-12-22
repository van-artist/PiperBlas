#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <ctime>
#include <Eigen/Sparse>

#include "pi_blas.hpp"
#include "pi_csr.hpp"
#include "pi_type.hpp"
#include "core/pi_config.hpp"
#include "core/test_utils.hpp"

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

    double pi_ms = time_avg_ms(warmup, iters, [&]()
                               { piSpMV(&A, x, y_pi); });
    double eigen_ms = time_avg_ms(warmup, iters, [&]()
                                  { y_eigen_map.noalias() = Aeigen * x_map; });

    Result out{};
    out.name = bin_path;
    out.m = m;
    out.n = n;
    out.nnz = nnz;
    const double ops = 2.0 * (double)nnz;
    out.gflops_pi = ops / (pi_ms * 1e-3) * 1e-9;
    out.gflops_eigen = ops / (eigen_ms * 1e-3) * 1e-9;

    free(x);
    free(y_pi);
    free(y_eigen);
    csr_destroy(&A);
    return out;
}

int main(int argc, char **argv)
{

    if (argc < 2)
    {
        std::fprintf(stderr, "用法: %s <matrix.bin|dir> [--warmup=2] [--iters=10]\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    int warmup = parse_int_flag(argc, argv, "--warmup=", 2);
    int iters = parse_int_flag(argc, argv, "--iters=", 10);

    std::vector<std::string> inputs = collect_inputs_or_exit(path, argv[0]);

    std::vector<Result> results;
    results.reserve(inputs.size());
    for (auto &p : inputs)
        results.push_back(run_case(p.c_str(), warmup, iters));

    TablePrinter table("==== SpMV CPU (Eigen vs pi) ====",
                       {"file", "m", "n", "nnz", "pi_GF/s", "eigen_GF/s"},
                       {TablePrinter::Align::Left, TablePrinter::Align::Right, TablePrinter::Align::Right,
                        TablePrinter::Align::Right, TablePrinter::Align::Right, TablePrinter::Align::Right});

    for (const auto &r : results)
    {
        table.add_row({
            basename_of(r.name),
            format_int64(r.m, 0),
            format_int64(r.n, 0),
            format_int64(r.nnz, 0),
            format_fixed(r.gflops_pi, 10, 3),
            format_fixed(r.gflops_eigen, 10, 3),
        });
    }

    table.print();

    return 0;
}
