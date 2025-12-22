#include <cstdio>

#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <string>
#include <type_traits>
#include <time.h>
#include <sys/time.h>

#include "pi_blas.hpp"
#include "pi_type.hpp"
#include "core/common.hpp"
#include "core/pi_config.hpp"
#include "core/test_utils.hpp"

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
        double blas_ms, blas_gflops;
        double v2_ms, v2_gflops, v2_err;
    };

    std::vector<Row> rows(n_scales);

    using PiFunc = piState (*)(T *, T *, T *, T, T, size_t, size_t, size_t);
    PiFunc v2 = std::is_same<T, double>::value ? (PiFunc)piGemmFp64 : (PiFunc)piGemmFp32;

    for (size_t t = 0; t < n_scales; ++t)
    {
        size_t N = Ns[t];
        size_t m = N, k = N, n = N;
        size_t asz = m * k;
        size_t bsz = k * n;
        size_t csz = m * n;

        T *A = (T *)aligned_alloc64(asz * sizeof(T));
        T *B = (T *)aligned_alloc64(bsz * sizeof(T));
        T *C0 = (T *)aligned_alloc64(csz * sizeof(T));
        T *Cblas = (T *)aligned_alloc64(csz * sizeof(T));
        T *C2 = (T *)aligned_alloc64(csz * sizeof(T));

        uint64_t seed = 1;
        fill_random(A, asz, &seed);
        fill_random(B, bsz, &seed);
        fill_random(C0, csz, &seed);

        for (size_t i = 0; i < csz; ++i)
        {
            Cblas[i] = C0[i];
            C2[i] = C0[i];
            C2[i] = C0[i];
        }

        double ops = 2.0 * (double)m * (double)k * (double)n;

        Row row{};
        row.N = N;
        row.blas_ms = time_avg_ms(0, 1, [&]()
                                  { host_blas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                                   (int)m, (int)n, (int)k,
                                                   alpha,
                                                   A, (int)k,
                                                   B, (int)n,
                                                   beta,
                                                   Cblas, (int)n); });
        row.blas_gflops = ops / (row.blas_ms * 1e-3) * 1e-9;

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

        row.v2_ms = time_avg_ms(0, 1, [&]()
                                { v2(A, B, C2, alpha, beta, m, k, n); });
        row.v2_gflops = ops / (row.v2_ms * 1e-3) * 1e-9;
        row.v2_err = eval_max_err(C2);

        rows[t] = row;

        free(A);
        free(B);
        free(C0);
        free(Cblas);
        free(C2);
    }

    TablePrinter table(
        std::string("==== CPU only ") + precision_name<T>() + " ====",
        {"N", "blas_ms", "blas_GF/s", "v2_ms", "v2_GF/s", "v2_err"},
        {TablePrinter::Align::Right, TablePrinter::Align::Right, TablePrinter::Align::Right,
         TablePrinter::Align::Right, TablePrinter::Align::Right, TablePrinter::Align::Right});

    for (const auto &r : rows)
    {
        table.add_row({
            format_int64((std::int64_t)r.N, 0),
            format_fixed(r.blas_ms, 10, 3),
            format_fixed(r.blas_gflops, 10, 3),
            format_fixed(r.v2_ms, 10, 3),
            format_fixed(r.v2_gflops, 10, 3),
            format_scientific(r.v2_err, 10, 3)});
    }

    table.print();
}

int main()
{
    run_cpu_test<float>();
    // run_cpu_test<double>();
    return 0;
}
