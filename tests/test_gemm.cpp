#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <type_traits>
#include <vector>
#include <assert.h>

#include "pi_blas.h"
#include "pi_type.h"
#include "pi_config.h"
#include "utils.h"
#include "cuda/cuda_kernels.h"

static double wall_now()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double cpu_now()
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void *xalloc(size_t n_bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n_bytes) != 0 || !p)
    {
        fprintf(stderr, "alloc fail\n");
        exit(1);
    }
    return p;
}

template <typename T>
static void fill_data(T *x, size_t n, uint64_t *seed)
{
    for (size_t i = 0; i < n; i++)
    {
        *seed = (*seed * 2862933555777941757ULL) + 3037000493ULL;
        double v = ((double)(*seed >> 11) / (double)(1ULL << 53)) * 2.0 - 1.0;
        x[i] = static_cast<T>(v);
    }
}

template <typename T>
static void host_blas_gemm(CBLAS_LAYOUT layout,
                           CBLAS_TRANSPOSE transA,
                           CBLAS_TRANSPOSE transB,
                           int m, int n, int k,
                           T alpha,
                           const T *A, int lda,
                           const T *B, int ldb,
                           T beta,
                           T *C, int ldc);

template <>
void host_blas_gemm<float>(CBLAS_LAYOUT layout,
                           CBLAS_TRANSPOSE transA,
                           CBLAS_TRANSPOSE transB,
                           int m, int n, int k,
                           float alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           float beta,
                           float *C, int ldc)
{
    cblas_sgemm(layout, transA, transB,
                m, n, k,
                alpha, A, lda,
                B, ldb,
                beta, C, ldc);
}

template <>
void host_blas_gemm<double>(CBLAS_LAYOUT layout,
                            CBLAS_TRANSPOSE transA,
                            CBLAS_TRANSPOSE transB,
                            int m, int n, int k,
                            double alpha,
                            const double *A, int lda,
                            const double *B, int ldb,
                            double beta,
                            double *C, int ldc)
{
    cblas_dgemm(layout, transA, transB,
                m, n, k,
                alpha, A, lda,
                B, ldb,
                beta, C, ldc);
}

template <typename T>
static cublasStatus_t device_blas_gemm(cublasHandle_t handle,
                                       cublasOperation_t transA,
                                       cublasOperation_t transB,
                                       int m, int n, int k,
                                       const T *alpha,
                                       const T *A, int lda,
                                       const T *B, int ldb,
                                       const T *beta,
                                       T *C, int ldc);

template <>
cublasStatus_t device_blas_gemm<float>(cublasHandle_t handle,
                                       cublasOperation_t transA,
                                       cublasOperation_t transB,
                                       int m, int n, int k,
                                       const float *alpha,
                                       const float *A, int lda,
                                       const float *B, int ldb,
                                       const float *beta,
                                       float *C, int ldc)
{
    return cublasSgemm(handle, transA, transB,
                       m, n, k,
                       alpha, A, lda,
                       B, ldb,
                       beta, C, ldc);
}

template <>
cublasStatus_t device_blas_gemm<double>(cublasHandle_t handle,
                                        cublasOperation_t transA,
                                        cublasOperation_t transB,
                                        int m, int n, int k,
                                        const double *alpha,
                                        const double *A, int lda,
                                        const double *B, int ldb,
                                        const double *beta,
                                        double *C, int ldc)
{
    return cublasDgemm(handle, transA, transB,
                       m, n, k,
                       alpha, A, lda,
                       B, ldb,
                       beta, C, ldc);
}

template <typename T>
static const char *precision_name();

template <>
const char *precision_name<float>()
{
    return "fp32";
}

template <>
const char *precision_name<double>()
{
    return "fp64";
}

template <typename T>
static piState my_cuda_gemm(int m, int k, int n,
                            T *dA, T *dB, T *dC,
                            T alpha, T beta);

template <>
piState my_cuda_gemm<float>(int m, int k, int n,
                            float *dA, float *dB, float *dC,
                            float alpha, float beta)
{
    return piCudaGemmFp32(dA, dB, dC, alpha, beta, m, k, n);
}

template <>
piState my_cuda_gemm<double>(int m, int k, int n,
                             double *dA, double *dB, double *dC,
                             double alpha, double beta)
{
    return piCudaGemmFp64(dA, dB, dC, alpha, beta, m, k, n);
}

template <typename T>
static piState my_cuda_gemm_v2(int m, int k, int n,
                               T *dA, T *dB, T *dC,
                               T alpha, T beta);

template <>
piState my_cuda_gemm_v2<float>(int m, int k, int n,
                               float *dA, float *dB, float *dC,
                               float alpha, float beta)
{
    return piCudaGemmFp32_v2(dA, dB, dC, alpha, beta, m, k, n);
}

template <>
piState my_cuda_gemm_v2<double>(int m, int k, int n,
                                double *dA, double *dB, double *dC,
                                double alpha, double beta)
{
    return piCudaGemmFp64_v2(dA, dB, dC, alpha, beta, m, k, n);
}

template <typename T>
static void run_precision(cublasHandle_t handle)
{
    const size_t Ns[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    const size_t n_scales = sizeof(Ns) / sizeof(Ns[0]);
    const T alpha = static_cast<T>(1.2);
    const T beta = static_cast<T>(0.8);

    struct Row
    {
        size_t N;
        size_t elems;

        // CPU: 只保留 v2 / v3
        double cpu_wall[2];
        double cpu_gflops[2];
        double cpu_max_err[2];
        double cpu_l2[2];

        // 自己的 CUDA 实现 v1
        double mycu_wall;
        double mycu_gflops;
        double mycu_max_err;
        double mycu_l2;

        // 自己的 CUDA 实现 v2
        double mycu_v2_wall;
        double mycu_v2_gflops;
        double mycu_v2_max_err;
        double mycu_v2_l2;

        // host BLAS
        double blas_wall;
        double blas_gflops;

        // cuBLAS
        double cu_wall;
        double cu_gflops;
        double cu_max_err;
        double cu_l2;
    };

    std::vector<Row> rows(n_scales);

    // CPU 版本：只保留 v2 / v3
    using PiFunc = piState (*)(T *, T *, T *, T, T, size_t, size_t, size_t);

    PiFunc funcs[2] = {
        std::is_same<T, double>::value ? (PiFunc)piGemmFp64_v2 : (PiFunc)piGemmFp32_v2,
        std::is_same<T, double>::value ? (PiFunc)piGemmFp64_v3 : (PiFunc)piGemmFp32_v3};

    for (size_t t = 0; t < n_scales; ++t)
    {
        size_t N = Ns[t];
        size_t m = N, k = N, n = N;
        size_t asz = m * k;
        size_t bsz = k * n;
        size_t csz = m * n;

        T *A = (T *)xalloc(asz * sizeof(T));
        T *B = (T *)xalloc(bsz * sizeof(T));
        T *Cref = (T *)xalloc(csz * sizeof(T));
        T *Cblas = (T *)xalloc(csz * sizeof(T));

        uint64_t seed = 1;
        fill_data(A, asz, &seed);
        fill_data(B, bsz, &seed);
        fill_data(Cref, csz, &seed);

        for (size_t i = 0; i < csz; ++i)
            Cblas[i] = Cref[i];

        // host BLAS 作为参考
        double w0 = wall_now();
        double c0 = cpu_now();
        (void)c0;
        host_blas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       (int)m, (int)n, (int)k,
                       alpha,
                       A, (int)k,
                       B, (int)n,
                       beta,
                       Cblas, (int)n);
        double c1 = cpu_now();
        (void)c1;
        double w1 = wall_now();

        double blas_wall = w1 - w0;
        double ops = 2.0 * (double)m * (double)k * (double)n;
        double blas_gflops = ops / blas_wall * 1e-9;

        // GPU 结果缓冲
        T *Ccu = (T *)xalloc(csz * sizeof(T));
        T *Cmy = (T *)xalloc(csz * sizeof(T));
        T *Cmy2 = (T *)xalloc(csz * sizeof(T));

        for (size_t i = 0; i < csz; ++i)
        {
            Ccu[i] = Cref[i];
            Cmy[i] = Cref[i];
            Cmy2[i] = Cref[i];
        }

        T *dA = nullptr;
        T *dB = nullptr;
        T *dC = nullptr;
        CHECK_CUDA(cudaMalloc(&dA, asz * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&dB, bsz * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&dC, csz * sizeof(T)));

        CHECK_CUDA(cudaMemcpy(dA, A, asz * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B, bsz * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dC, Ccu, csz * sizeof(T), cudaMemcpyHostToDevice));

        // cuBLAS
        double w2 = wall_now();
        device_blas_gemm(handle,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         (int)n, (int)m, (int)k,
                         &alpha,
                         dB, (int)n,
                         dA, (int)k,
                         &beta,
                         dC, (int)n);
        CHECK_CUDA(cudaDeviceSynchronize());
        double w3 = wall_now();

        CHECK_CUDA(cudaMemcpy(Ccu, dC, csz * sizeof(T), cudaMemcpyDeviceToHost));

        double cu_wall = w3 - w2;
        double cu_gflops = ops / cu_wall * 1e-9;

        double cu_max_err = 0.0, cu_l2 = 0.0;
        for (size_t i = 0; i < csz; ++i)
        {
            double d = (double)Ccu[i] - (double)Cblas[i];
            if (fabs(d) > cu_max_err)
                cu_max_err = fabs(d);
            cu_l2 += d * d;
        }
        cu_l2 = sqrt(cu_l2);

        // 自己的 CUDA 实现 v1
        CHECK_CUDA(cudaMemcpy(dC, Cmy, csz * sizeof(T), cudaMemcpyHostToDevice));

        double w4 = wall_now();
        piState st = my_cuda_gemm<T>((int)m, (int)k, (int)n, dA, dB, dC, alpha, beta);
        (void)st;
        CHECK_CUDA(cudaDeviceSynchronize());
        double w5 = wall_now();

        CHECK_CUDA(cudaMemcpy(Cmy, dC, csz * sizeof(T), cudaMemcpyDeviceToHost));

        double mycu_wall = w5 - w4;
        double mycu_gflops = ops / mycu_wall * 1e-9;

        double mycu_max_err = 0.0, mycu_l2 = 0.0;
        for (size_t i = 0; i < csz; ++i)
        {
            double d = (double)Cmy[i] - (double)Cblas[i];
            if (fabs(d) > mycu_max_err)
                mycu_max_err = fabs(d);
            mycu_l2 += d * d;
        }
        mycu_l2 = sqrt(mycu_l2);

        // 自己的 CUDA 实现 v2
        CHECK_CUDA(cudaMemcpy(dC, Cmy2, csz * sizeof(T), cudaMemcpyHostToDevice));

        double w6 = wall_now();
        piState st2 = my_cuda_gemm_v2<T>((int)m, (int)k, (int)n, dA, dB, dC, alpha, beta);
        (void)st2;
        CHECK_CUDA(cudaDeviceSynchronize());
        double w7 = wall_now();

        CHECK_CUDA(cudaMemcpy(Cmy2, dC, csz * sizeof(T), cudaMemcpyDeviceToHost));

        double mycu_v2_wall = w7 - w6;
        double mycu_v2_gflops = ops / mycu_v2_wall * 1e-9;

        double mycu_v2_max_err = 0.0, mycu_v2_l2 = 0.0;
        for (size_t i = 0; i < csz; ++i)
        {
            double d = (double)Cmy2[i] - (double)Cblas[i];
            if (fabs(d) > mycu_v2_max_err)
                mycu_v2_max_err = fabs(d);
            mycu_v2_l2 += d * d;
        }
        mycu_v2_l2 = sqrt(mycu_v2_l2);

        Row row{};
        row.N = N;
        row.elems = csz;
        row.blas_wall = blas_wall;
        row.blas_gflops = blas_gflops;
        row.cu_wall = cu_wall;
        row.cu_gflops = cu_gflops;
        row.cu_max_err = cu_max_err;
        row.cu_l2 = cu_l2;
        row.mycu_wall = mycu_wall;
        row.mycu_gflops = mycu_gflops;
        row.mycu_max_err = mycu_max_err;
        row.mycu_l2 = mycu_l2;
        row.mycu_v2_wall = mycu_v2_wall;
        row.mycu_v2_gflops = mycu_v2_gflops;
        row.mycu_v2_max_err = mycu_v2_max_err;
        row.mycu_v2_l2 = mycu_v2_l2;

        // CPU v2 / v3
        for (int v = 0; v < 2; ++v)
        {
            T *C = (T *)xalloc(csz * sizeof(T));
            for (size_t i = 0; i < csz; ++i)
                C[i] = Cref[i];

            double wv0 = wall_now();
            funcs[v](A, B, C, alpha, beta, m, k, n);
            double wv1 = wall_now();

            row.cpu_wall[v] = wv1 - wv0;
            row.cpu_gflops[v] = ops / row.cpu_wall[v] * 1e-9;

            double max_err = 0.0, l2 = 0.0;
            for (size_t i = 0; i < csz; ++i)
            {
                double d = (double)C[i] - (double)Cblas[i];
                if (fabs(d) > max_err)
                    max_err = fabs(d);
                l2 += d * d;
            }
            row.cpu_max_err[v] = max_err;
            row.cpu_l2[v] = sqrt(l2);

            free(C);
        }

        rows[t] = row;

        free(A);
        free(B);
        free(Cref);
        free(Cblas);
        free(Ccu);
        free(Cmy);
        free(Cmy2);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    printf("==== %s ====\n", precision_name<T>());
    printf("%8s %8s | %10s %10s %10s %10s %10s %10s | %10s %10s %10s %10s %10s\n",
           "N", "elems",
           "v2_GF/s", "v3_GF/s",
           "myCU1_GF/s", "myCU2_GF/s",
           "blas_GF/s", "cuBlas_GF/s",
           "v2_err", "v3_err",
           "myCU1_err", "myCU2_err", "cuBlas_err");

    for (size_t i = 0; i < rows.size(); ++i)
    {
        const Row &r = rows[i];
        printf("%8zu %8zu | %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f | "
               "%10.3e %10.3e %10.3e %10.3e %10.3e\n",
               r.N, r.elems,
               r.cpu_gflops[0], r.cpu_gflops[1],   // v2, v3
               r.mycu_gflops, r.mycu_v2_gflops,    // myCUDA v1, v2
               r.blas_gflops, r.cu_gflops,         // host BLAS, cuBLAS
               r.cpu_max_err[0], r.cpu_max_err[1], // v2, v3 err
               r.mycu_max_err, r.mycu_v2_max_err,  // myCUDA v1, v2 err
               r.cu_max_err);                      // cuBLAS err
    }
}

int main()
{
    config_init();

    cublasHandle_t handle;
    cublasCreate(&handle);

    run_precision<double>(handle);
    run_precision<float>(handle);

    cublasDestroy(handle);
    return 0;
}
