#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <type_traits>
#include <time.h>
#include <sys/time.h>

#include "pi_blas.h"
#include "pi_type.h"
#include "utils.h"
#include "pi_config.h"

static double wall_now()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
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
static const char *precision_name();

template <>
const char *precision_name<float>() { return "fp32"; }

template <>
const char *precision_name<double>() { return "fp64"; }

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
static void run_cpu_test()
{
    const size_t Ns[] = {
        64, 96, 128,
        160, 192, 224, 256,
        320, 384, 448, 512,
        640, 768, 896, 1024,
        1280, 1536, 1792, 2048,
        2560, 3072, 3584, 4096,
        5120, 6144, 7168, 8192};
    const size_t n_scales = sizeof(Ns) / sizeof(Ns[0]);

    const T alpha = static_cast<T>(1.2);
    const T beta = static_cast<T>(0.8);

    struct Row
    {
        size_t N;
        double blas_wall, blas_gflops;
        double v2_wall, v2_gflops, v2_err;
        double v3_wall, v3_gflops, v3_err;
    };

    std::vector<Row> rows(n_scales);

    using PiFunc = piState (*)(T *, T *, T *, T, T, size_t, size_t, size_t);
    PiFunc v2 = std::is_same<T, double>::value ? (PiFunc)piGemmFp64_v2 : (PiFunc)piGemmFp32_v2;
    PiFunc v3 = std::is_same<T, double>::value ? (PiFunc)piGemmFp64_v3 : (PiFunc)piGemmFp32_v3;

    for (size_t t = 0; t < n_scales; ++t)
    {
        size_t N = Ns[t];
        size_t m = N, k = N, n = N;
        size_t asz = m * k;
        size_t bsz = k * n;
        size_t csz = m * n;

        T *A = (T *)xalloc(asz * sizeof(T));
        T *B = (T *)xalloc(bsz * sizeof(T));
        T *C0 = (T *)xalloc(csz * sizeof(T));
        T *Cblas = (T *)xalloc(csz * sizeof(T));
        T *C2 = (T *)xalloc(csz * sizeof(T));
        T *C3 = (T *)xalloc(csz * sizeof(T));

        uint64_t seed = 1;
        fill_data(A, asz, &seed);
        fill_data(B, bsz, &seed);
        fill_data(C0, csz, &seed);

        for (size_t i = 0; i < csz; ++i)
        {
            Cblas[i] = C0[i];
            C2[i] = C0[i];
            C3[i] = C0[i];
        }

        double ops = 2.0 * (double)m * (double)k * (double)n;

        double w0 = wall_now();
        host_blas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       (int)m, (int)n, (int)k,
                       alpha,
                       A, (int)k,
                       B, (int)n,
                       beta,
                       Cblas, (int)n);
        double w1 = wall_now();

        Row row{};
        row.N = N;
        row.blas_wall = w1 - w0;
        row.blas_gflops = ops / row.blas_wall * 1e-9;

        auto eval_max_err = [&](const T *X) -> double
        {
            double max_err = 0.0;
            for (size_t i = 0; i < csz; ++i)
            {
                double d = (double)X[i] - (double)Cblas[i];
                if (fabs(d) > max_err)
                    max_err = fabs(d);
            }
            return max_err;
        };

        double w2 = wall_now();
        v2(A, B, C2, alpha, beta, m, k, n);
        double w3 = wall_now();
        row.v2_wall = w3 - w2;
        row.v2_gflops = ops / row.v2_wall * 1e-9;
        row.v2_err = eval_max_err(C2);

        double w4 = wall_now();
        v3(A, B, C3, alpha, beta, m, k, n);
        double w5 = wall_now();
        row.v3_wall = w5 - w4;
        row.v3_gflops = ops / row.v3_wall * 1e-9;
        row.v3_err = eval_max_err(C3);

        rows[t] = row;

        free(A);
        free(B);
        free(C0);
        free(Cblas);
        free(C2);
        free(C3);
    }

    printf("==== CPU only %s ====\n", precision_name<T>());
    printf("%8s | %10s %10s | %10s %10s %10s | %10s %10s %10s\n",
           "N",
           "blas_ms", "blas_GF/s",
           "v2_ms", "v2_GF/s", "v2_err",
           "v3_ms", "v3_GF/s", "v3_err");

    for (const auto &r : rows)
    {
        printf("%8zu | %10.3f %10.3f | %10.3f %10.3f %10.3e | %10.3f %10.3f %10.3e\n",
               r.N,
               r.blas_wall * 1e3, r.blas_gflops,
               r.v2_wall * 1e3, r.v2_gflops, r.v2_err,
               r.v3_wall * 1e3, r.v3_gflops, r.v3_err);
    }
}

int main()
{
    config_init();
    run_cpu_test<float>();
    // run_cpu_test<double>();
    return 0;
}
