#if !defined(PIPER_HAVE_CUDA) || !PIPER_HAVE_CUDA
#include <cstdio>

int main()
{
    std::fprintf(stderr, "CUDA support is disabled.\n");
    return 0;
}

#else
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>

#include "pi_blas.hpp"
#include "pi_csr.hpp"
#include "pi_type.hpp"
#include "core/pi_config.hpp"
#include "core/test_utils.hpp"
#include "cuda/cuda_common.cuh"
#include "cuda/cuda_test_utils.cuh"
#include "cuda/cuda_kernels.cuh"

struct Result
{
    std::string name;
    int m, n, nnz;
    double gflops_fp32;
    double gflops_fp64;
};

static Result run_case(const char *bin_path, int warmup, int iters)
{
    pi_csr A;
    if (csr_from_bin(bin_path, &A) != piSuccess)
    {
        std::fprintf(stderr, "csr_from_bin 读取失败: %s\n", bin_path);
        std::exit(1);
    }

    const int m = A.n_rows;
    const int n = A.n_cols;
    const int nnz = A.nnz;

    double *x = (double *)aligned_alloc64((size_t)n * sizeof(double));
    double *y = (double *)aligned_alloc64((size_t)m * sizeof(double));

    uint64_t seed = 1;
    fill_random_double(x, (size_t)n, &seed);

    int *d_row_ptr = nullptr;
    int *d_col_idx = nullptr;
    double *d_values = nullptr;
    double *d_x = nullptr;
    double *d_y = nullptr;

    CHECK_CUDA(cudaMalloc(&d_row_ptr, ((size_t)m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_idx, (size_t)nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, (size_t)nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x, (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, (size_t)m * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_row_ptr, A.row_ptr, ((size_t)m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_idx, A.col_idx, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, A.values, (size_t)nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));

    pi_csr A_dev = A;
    A_dev.row_ptr = d_row_ptr;
    A_dev.col_idx = d_col_idx;
    A_dev.values = d_values;

    for (int i = 0; i < warmup; ++i)
    {
        (void)pi_cuda_spmv_fp64(&A_dev, d_x, d_y);
        (void)pi_cuda_spmv_fp32(&A_dev, d_x, d_y);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms64 = cuda_time_avg_ms([&]() { (void)pi_cuda_spmv_fp64(&A_dev, d_x, d_y); }, warmup, iters);
    float ms32 = cuda_time_avg_ms([&]() { (void)pi_cuda_spmv_fp32(&A_dev, d_x, d_y); }, warmup, iters);

    Result out{};
    out.name = bin_path;
    out.m = m;
    out.n = n;
    out.nnz = nnz;
    const double ops = 2.0 * (double)nnz;
    out.gflops_fp64 = ops / (ms64 * 1e-3) * 1e-9;
    out.gflops_fp32 = ops / (ms32 * 1e-3) * 1e-9;

    CHECK_CUDA(cudaFree(d_row_ptr));
    CHECK_CUDA(cudaFree(d_col_idx));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    free(x);
    free(y);
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

    std::printf("==== SpMV CUDA (timing only) ====\n");
    std::printf("%32s | %8s %8s %10s | %10s %10s\n", "file", "m", "n", "nnz", "fp64_GF/s", "fp32_GF/s");

    for (const auto &r : results)
    {
        const char *name = r.name.c_str();
        const char *base = std::strrchr(name, '/');
        base = base ? base + 1 : name;

        std::printf("%32s | %8d %8d %10d | %10.3f %10.3f\n",
                    base, r.m, r.n, r.nnz,
                    r.gflops_fp64, r.gflops_fp32);
    }

    return 0;
}
#endif
