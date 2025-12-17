#if !defined(PIPER_HAVE_CUDA) || !PIPER_HAVE_CUDA || !defined(PIPER_HAVE_MKL) || !PIPER_HAVE_MKL
#include <cstdio>

int main()
{
    std::fprintf(stderr, "CUDA or MKL support is disabled.\n");
    return 0;
}

#else
#include <mkl.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <string>
#include <string.h>
#include <limits>
#include <time.h>

#include <dirent.h>
#include <sys/stat.h>

#include "pi_blas.h"
#include "pi_csr.h"
#include "pi_type.h"
#include "core/pi_config.h"
#include "core/common.h"
#include "cuda/cuda_kernels.h"

piState pi_cuda_spmv_fp32(const pi_csr *__restrict A, double *__restrict x, double *__restrict y);
piState pi_cuda_spmv_fp64(const pi_csr *__restrict A, double *__restrict x, double *__restrict y);

static void *aligned_alloc64(size_t bytes)
{
    void *p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0 || !p)
    {
        fprintf(stderr, "alloc fail\n");
        exit(1);
    }
    return p;
}

static void fill_random_double(double *x, size_t n, uint64_t *seed)
{
    for (size_t i = 0; i < n; i++)
    {
        *seed = (*seed * 2862933555777941757ULL) + 3037000493ULL;
        double v = ((double)(*seed >> 11) / (double)(1ULL << 53)) * 2.0 - 1.0;
        x[i] = v;
    }
}

template <class F>
static float time_avg_ms(F launch, int warmup, int iters)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i)
        launch();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        launch();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / (float)iters;
}

static inline double rel_pct(double custom, double ref)
{
    return ref > 0.0 ? (custom / ref) * 100.0 : 0.0;
}

static void compute_err(const double *a, const double *b, int m, double *max_err, double *l2)
{
    double me = 0.0, s2 = 0.0;
    for (int i = 0; i < m; ++i)
    {
        double d = fabs(a[i] - b[i]);
        if (d > me)
            me = d;
        s2 += d * d;
    }
    *max_err = me;
    *l2 = sqrt(s2);
}

enum ShowMask : int
{
    SHOW_ABS = 1 << 0,
    SHOW_REL = 1 << 1,
};

static int parse_show_mask(int argc, char **argv)
{
    int mask = SHOW_ABS | SHOW_REL;
    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        if (strncmp(a, "--show=", 7) == 0)
        {
            mask = 0;
            std::string s(a + 7);
            for (auto &c : s)
                if (c == ',')
                    c = '+';
            if (s.find("abs") != std::string::npos)
                mask |= SHOW_ABS;
            if (s.find("rel") != std::string::npos)
                mask |= SHOW_REL;
            if (mask == 0)
                mask = SHOW_ABS | SHOW_REL;
        }
    }
    return mask;
}

static int parse_int_flag(int argc, char **argv, const char *key, int defv)
{
    const size_t klen = strlen(key);
    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        if (strncmp(a, key, klen) == 0 && a[klen] == '=')
            return atoi(a + klen + 1);
    }
    return defv;
}

static bool is_directory(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        return false;
    return S_ISDIR(st.st_mode);
}

static bool is_regular_file(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        return false;
    return S_ISREG(st.st_mode);
}

static std::string join_path(const std::string &dir, const std::string &name)
{
    if (dir.empty())
        return name;
    if (dir.back() == '/')
        return dir + name;
    return dir + "/" + name;
}

static std::vector<std::string> list_files_in_dir(const char *dir_path)
{
    std::vector<std::string> files;

    DIR *d = opendir(dir_path);
    if (!d)
        return files;

    while (true)
    {
        struct dirent *ent = readdir(d);
        if (!ent)
            break;

        const char *name = ent->d_name;
        if (!name || strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
            continue;

        std::string full = join_path(dir_path, name);
        if (is_regular_file(full.c_str()))
            files.push_back(full);
    }

    closedir(d);
    std::sort(files.begin(), files.end());
    return files;
}

struct Result
{
    std::string name;
    int m, n, nnz;
    double gflops_pi, gflops_mkl, gflops_gpu64, gflops_gpu32;
    double maxerr_pi, l2_pi;
    double maxerr_gpu64, l2_gpu64;
    double maxerr_gpu32, l2_gpu32;
};

static Result run_case(const char *bin_path, int warmup, int iters)
{
    pi_csr A;
    piState st = csr_from_bin(bin_path, &A);
    if (st != piSuccess)
    {
        fprintf(stderr, "csr_from_bin 读取失败: %s\n", bin_path);
        exit(1);
    }

    int m = A.n_rows;
    int n = A.n_cols;
    int nnz = A.nnz;

    double *x = (double *)aligned_alloc64((size_t)n * sizeof(double));
    double *y_pi = (double *)aligned_alloc64((size_t)m * sizeof(double));
    double *y_mkl = (double *)aligned_alloc64((size_t)m * sizeof(double));
    double *y_gpu64 = (double *)aligned_alloc64((size_t)m * sizeof(double));
    double *y_gpu32 = (double *)aligned_alloc64((size_t)m * sizeof(double));

    uint64_t seed = 1;
    fill_random_double(x, (size_t)n, &seed);

    MKL_INT m_mkl = (MKL_INT)m;
    MKL_INT n_mkl = (MKL_INT)n;

    std::vector<MKL_INT> mkl_row_ptr((size_t)m + 1);
    std::vector<MKL_INT> mkl_col_idx((size_t)nnz);
    std::vector<double> mkl_values((size_t)nnz);

    for (int i = 0; i <= m; ++i)
        mkl_row_ptr[(size_t)i] = (MKL_INT)A.row_ptr[i];
    for (int j = 0; j < nnz; ++j)
    {
        mkl_col_idx[(size_t)j] = (MKL_INT)A.col_idx[j];
        mkl_values[(size_t)j] = A.values[j];
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
        exit(1);
    }

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_optimize(Amkl);

    int *d_row_ptr = nullptr;
    int *d_col_idx = nullptr;
    double *d_values = nullptr;
    double *d_x = nullptr;
    double *d_y = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_row_ptr, ((size_t)m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_col_idx, (size_t)nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_values, (size_t)nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&d_x, (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&d_y, (size_t)m * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_row_ptr, A.row_ptr, ((size_t)m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_idx, A.col_idx, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, A.values, (size_t)nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));

    pi_csr A_dev = A;
    A_dev.row_ptr = d_row_ptr;
    A_dev.col_idx = d_col_idx;
    A_dev.values = d_values;

    memset(y_pi, 0, (size_t)m * sizeof(double));
    memset(y_mkl, 0, (size_t)m * sizeof(double));
    memset(y_gpu64, 0, (size_t)m * sizeof(double));
    memset(y_gpu32, 0, (size_t)m * sizeof(double));

    piSpMV(&A, x, y_pi);

    status = mkl_sparse_d_mv(
        SPARSE_OPERATION_NON_TRANSPOSE,
        1.0,
        Amkl,
        descr,
        x,
        0.0,
        y_mkl);
    if (status != SPARSE_STATUS_SUCCESS)
    {
        fprintf(stderr, "mkl_sparse_d_mv 失败, status = %d\n", status);
        exit(1);
    }

    (void)pi_cuda_spmv_fp64(&A_dev, d_x, d_y);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(y_gpu64, d_y, (size_t)m * sizeof(double), cudaMemcpyDeviceToHost));

    (void)pi_cuda_spmv_fp32(&A_dev, d_x, d_y);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(y_gpu32, d_y, (size_t)m * sizeof(double), cudaMemcpyDeviceToHost));

    Result out{};
    out.name = bin_path;
    out.m = m;
    out.n = n;
    out.nnz = nnz;

    compute_err(y_pi, y_mkl, m, &out.maxerr_pi, &out.l2_pi);
    compute_err(y_gpu64, y_mkl, m, &out.maxerr_gpu64, &out.l2_gpu64);
    compute_err(y_gpu32, y_mkl, m, &out.maxerr_gpu32, &out.l2_gpu32);

    const double ops = 2.0 * (double)nnz;

    for (int i = 0; i < warmup; ++i)
    {
        piSpMV(&A, x, y_pi);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, Amkl, descr, x, 0.0, y_mkl);
        (void)pi_cuda_spmv_fp64(&A_dev, d_x, d_y);
        (void)pi_cuda_spmv_fp32(&A_dev, d_x, d_y);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    auto time_cpu_s = [&](auto fn)
    {
        double t0 = 0.0, t1 = 0.0;
        t0 = (double)clock() / (double)CLOCKS_PER_SEC;
        for (int i = 0; i < iters; ++i)
            fn();
        t1 = (double)clock() / (double)CLOCKS_PER_SEC;
        return (t1 - t0) / (double)iters;
    };

    double pi_s = time_cpu_s([&]()
                             { piSpMV(&A, x, y_pi); });
    double mkl_s = time_cpu_s([&]()
                              { mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, Amkl, descr, x, 0.0, y_mkl); });

    float gpu64_ms = time_avg_ms([&]()
                                 { (void)pi_cuda_spmv_fp64(&A_dev, d_x, d_y); }, warmup, iters);
    float gpu32_ms = time_avg_ms([&]()
                                 { (void)pi_cuda_spmv_fp32(&A_dev, d_x, d_y); }, warmup, iters);

    out.gflops_pi = ops / pi_s * 1e-9;
    out.gflops_mkl = ops / mkl_s * 1e-9;
    out.gflops_gpu64 = ops / (gpu64_ms * 1e-3) * 1e-9;
    out.gflops_gpu32 = ops / (gpu32_ms * 1e-3) * 1e-9;

    CHECK_CUDA(cudaFree(d_row_ptr));
    CHECK_CUDA(cudaFree(d_col_idx));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    mkl_sparse_destroy(Amkl);

    free(x);
    free(y_pi);
    free(y_mkl);
    free(y_gpu64);
    free(y_gpu32);

    csr_destroy(&A);

    return out;
}

int main(int argc, char **argv)
{
    config_init();

    if (argc < 2)
    {
        fprintf(stderr, "用法: %s <matrix.bin|dir> [--warmup=2] [--iters=10] [--show=abs,rel]\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    int warmup = parse_int_flag(argc, argv, "--warmup", 2);
    int iters = parse_int_flag(argc, argv, "--iters", 10);
    int show = parse_show_mask(argc, argv);

    std::vector<std::string> inputs;
    if (is_directory(path))
        inputs = list_files_in_dir(path);
    else
        inputs.push_back(path);

    if (inputs.empty())
    {
        fprintf(stderr, "没有找到输入文件: %s\n", path);
        return 1;
    }

    std::vector<Result> results;
    results.reserve(inputs.size());
    for (auto &p : inputs)
        results.push_back(run_case(p.c_str(), warmup, iters));

    printf("==== SpMV (timing only) ====\n");

    printf("%32s | %8s %8s %10s", "file", "m", "n", "nnz");
    if (show & SHOW_ABS)
        printf(" | %10s %10s %10s %10s", "pi_GF/s", "mkl_GF/s", "g64_GF/s", "g32_GF/s");
    if (show & SHOW_REL)
        printf(" | %10s %10s %10s", "pi_%mkl", "g64_%mkl", "g32_%mkl");
    printf("\n");

    for (const auto &r : results)
    {
        const char *name = r.name.c_str();
        const char *base = strrchr(name, '/');
        base = base ? base + 1 : name;

        printf("%32s | %8d %8d %10d", base, r.m, r.n, r.nnz);

        if (show & SHOW_ABS)
            printf(" | %10.3f %10.3f %10.3f %10.3f",
                   r.gflops_pi, r.gflops_mkl, r.gflops_gpu64, r.gflops_gpu32);

        if (show & SHOW_REL)
            printf(" | %10.2f %10.2f %10.2f",
                   rel_pct(r.gflops_pi, r.gflops_mkl),
                   rel_pct(r.gflops_gpu64, r.gflops_mkl),
                   rel_pct(r.gflops_gpu32, r.gflops_mkl));

        printf("\n");
    }

    return 0;
}
#endif
