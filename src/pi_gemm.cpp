#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <pthread.h>
#include "utils.h"
#include "pi_blas.h"
#include "pi_type.h"
#include "pi_config.h"

#define BLOCK_SIZE 128

template <typename T>
struct pi_gemm_arg
{
    T *A;
    T *B;
    T *C;
    T alpha;
    T beta;
    size_t m;
    size_t k;
    size_t n;
};

template <typename T>
struct pi_gemm_arg_v2
{
    T *A;
    T *B;
    T *C;
    T alpha;
    T beta;
    size_t mb;
    size_t nb;
    size_t kb;
    size_t strid_a;
    size_t strid_b;
    size_t strid_c;
    size_t c_idx_i;
    size_t c_idx_j;
    size_t k; // 全局k
};

typedef struct
{
    piState state;
} pi_gemm_retval;

// 按M纬切分下单个线程的处理
template <typename T>
static void *pi_gemm_m(void *arg_)
{
    pi_gemm_arg<T> *arg = (pi_gemm_arg<T> *)arg_;
    size_t offset_a = 0;
    size_t offset_b = 0;
    size_t offset_c = 0;

    for (size_t i = 0; i < arg->m; i++)
    {
        T *c_row = arg->C + i * arg->n;
        for (size_t j = 0; j < arg->n; j++)
        {
            c_row[j] *= arg->beta;
        }
    }

    for (size_t i = 0; i < arg->m; i++)
    {
        // 取A的第i行
        for (size_t l = 0; l < arg->k; l++)
        {
            // B的第l行
            offset_a = i * arg->k + l;
            for (size_t j = 0; j < arg->n; j++)
            {
                // C的第i行的第j个元素做累加
                offset_b = l * arg->n + j;
                offset_c = i * arg->n + j;
                arg->C[offset_c] += arg->A[offset_a] * arg->B[offset_b] * arg->alpha;
            }
        }
    }

    pi_gemm_retval *retval = (pi_gemm_retval *)malloc(sizeof(pi_gemm_retval));
    if (!retval)
        return NULL;
    retval->state = piSuccess;
    return (void *)retval;
}

template <typename T>
static piState piGemmImpl(T *__restrict A, T *__restrict B, T *__restrict C, T alpha, T beta, size_t m, size_t k, size_t n)
{
    // A是m行k列
    // B是k行n列
    // C是m行n列
    // 按照M纬度划分任务
    int thread_num = config()->thread_num;
    // M特别小的时候，直接单线程计算，这个后续可以丰富调度策略
    if (thread_num <= 1 || m < (size_t)thread_num)
    {
        pi_gemm_arg<T> arg;
        arg.A = A;
        arg.B = B;
        arg.C = C;
        arg.alpha = alpha;
        arg.beta = beta;
        arg.m = m;
        arg.k = k;
        arg.n = n;
        pi_gemm_m<T>(&arg);
        return piSuccess;
    }

    // 准备参数
    pi_gemm_arg<T> *args = (pi_gemm_arg<T> *)malloc(sizeof(pi_gemm_arg<T>) * (size_t)thread_num);
    int single_m = (int)(m / (size_t)thread_num);
    int last_m = (int)(m % (size_t)thread_num);
    int has_last = last_m;
    for (int i = 0; i < thread_num; i++)
    {
        args[i].alpha = alpha;
        args[i].B = B;
        args[i].k = k;
        args[i].n = n;
        args[i].beta = beta;
        args[i].m = (size_t)single_m;
        if (has_last && i == thread_num - 1)
        {
            args[i].m = (size_t)single_m + (size_t)last_m;
        }
        size_t offset_a = (size_t)i * (size_t)single_m * k;
        size_t offset_c = (size_t)i * (size_t)single_m * n;
        args[i].A = A + offset_a;
        args[i].C = C + offset_c;
    }
    // 创建线程
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)thread_num);

    for (int i = 0; i < thread_num; i++)
    {
        pthread_create(&threads[i], NULL, pi_gemm_m<T>, &args[i]);
    }
    // 等待线程
    for (int i = 0; i < thread_num; i++)
    {
        void *retv = NULL;
        pthread_join(threads[i], &retv);
        if (retv)
        {
            pi_gemm_retval *r = (pi_gemm_retval *)retv;
            (void)r;
            free(r);
        }
    }

    free(args);
    free(threads);
    return piSuccess;
}

piState piGemmFp64(double *__restrict A, double *__restrict B, double *__restrict C, double alpha, double beta, size_t m, size_t k, size_t n)
{
    return piGemmImpl<double>(A, B, C, alpha, beta, m, k, n);
}

piState piGemmFp32(float *__restrict A, float *__restrict B, float *__restrict C, float alpha, float beta, size_t m, size_t k, size_t n)
{
    return piGemmImpl<float>(A, B, C, alpha, beta, m, k, n);
}

piState piGemm(double *__restrict A, double *__restrict B, double *__restrict C, double alpha, double beta, size_t m, size_t k, size_t n)
{
    return piGemmFp64(A, B, C, alpha, beta, m, k, n);
}

// 二维下A*B子块的gemm逻辑
template <typename T>
static void pi_gemm_block(T *__restrict A_block,
                          T *__restrict B_block,
                          T *__restrict C_block,
                          T alpha, T beta,
                          size_t mb, size_t kb, size_t nb,
                          size_t strid_a, size_t strid_b, size_t strid_c)
{
    if (beta != T(1))
    {
        for (size_t i = 0; i < mb; ++i)
        {
            T *Ci = C_block + i * strid_c;
            for (size_t j = 0; j < nb; ++j)
                Ci[j] *= beta;
        }
    }

    for (size_t i = 0; i < mb; ++i)
    {
        T *A_row = A_block + i * strid_a;
        T *C_row = C_block + i * strid_c;

        for (size_t l = 0; l < kb; ++l)
        {
            T a = alpha * A_row[l];
            const T *B_row = B_block + l * strid_b;

            for (size_t j = 0; j < nb; ++j)
            {
                C_row[j] += a * B_row[j];
            }
        }
    }
}

// 二维切分下的单个线程，对C的一个子块负责,也就是A的一行子块，B的一列子块
template <typename T>
static void *pi_gemm_mn(void *arg_)
{
    pi_gemm_arg_v2<T> *arg = (pi_gemm_arg_v2<T> *)arg_;

    T *A = arg->A;
    T *B = arg->B;
    T *C = arg->C;
    const T alpha = arg->alpha;
    const T beta0 = arg->beta;

    const size_t mb = arg->mb;
    const size_t nb = arg->nb;
    const size_t kb = arg->kb;

    const size_t strid_a = arg->strid_a;
    const size_t strid_b = arg->strid_b;
    const size_t strid_c = arg->strid_c;

    const size_t c_idx_i = arg->c_idx_i;
    const size_t c_idx_j = arg->c_idx_j;
    const size_t k = arg->k;

    T *C_block = C + c_idx_i * strid_c + c_idx_j;

    T beta_used = beta0;
    for (size_t offset_k = 0; offset_k < k; offset_k += kb)
    {
        size_t cur_kb = kb;
        if (offset_k + cur_kb > k)
            // 最后一块
            cur_kb = k - offset_k;

        T *A_block = A + c_idx_i * strid_a + offset_k;
        T *B_block = B + offset_k * strid_b + c_idx_j;

        pi_gemm_block(A_block, B_block, C_block,
                      alpha, beta_used,
                      mb, cur_kb, nb,
                      strid_a, strid_b, strid_c);

        beta_used = T(1);
    }

    return NULL;
}

template <typename T>
static piState piGemmV2Impl(T *A, T *B, T *C,
                            T alpha, T beta,
                            size_t m, size_t k, size_t n)
{
    const int thread_num_cfg = config()->thread_num;
    const int thread_num = (thread_num_cfg > 0) ? thread_num_cfg : 1;

    const size_t MB = BLOCK_SIZE;
    const size_t NB = BLOCK_SIZE;
    const size_t KB = BLOCK_SIZE;

    // 线程为1或者形状小的时候的特殊处理
    if (m == 0 || n == 0 || k == 0)
        return piSuccess;

    if (thread_num == 1)
    {
        for (size_t i = 0; i < m; i += MB)
        {
            const size_t mb = (i + MB <= m) ? MB : (m - i);
            for (size_t j = 0; j < n; j += NB)
            {
                const size_t nb = (j + NB <= n) ? NB : (n - j);

                pi_gemm_arg_v2<T> arg;
                arg.A = A;
                arg.B = B;
                arg.C = C;
                arg.alpha = alpha;
                arg.beta = beta;
                arg.mb = mb;
                arg.nb = nb;
                arg.kb = KB;
                arg.strid_a = k;
                arg.strid_b = n;
                arg.strid_c = n;
                arg.c_idx_i = i;
                arg.c_idx_j = j;
                arg.k = k;

                (void)pi_gemm_mn<T>(&arg);
            }
        }
        return piSuccess;
    }

    // 初始化线程池和参数池
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)thread_num);
    pi_gemm_arg_v2<T> *args = (pi_gemm_arg_v2<T> *)malloc(sizeof(pi_gemm_arg_v2<T>) * (size_t)thread_num);
    if (!threads || !args)
    {
        free(threads);
        free(args);
        return piDataInvalid;
    }

    int launched = 0;
    for (size_t i = 0; i < m; i += MB)
    {
        const size_t mb = (i + MB <= m) ? MB : (m - i);
        for (size_t j = 0; j < n; j += NB)
        {
            const size_t nb = (j + NB <= n) ? NB : (n - j);

            // 准备本块参数
            pi_gemm_arg_v2<T> *arg = &args[launched];
            arg->A = A;
            arg->B = B;
            arg->C = C;
            arg->alpha = alpha;
            arg->beta = beta;
            arg->mb = mb;
            arg->nb = nb;
            arg->kb = KB;
            arg->strid_a = k;
            arg->strid_b = n;
            arg->strid_c = n;
            arg->c_idx_i = i;
            arg->c_idx_j = j;
            arg->k = k;

            // 启动一个线程执行该子块处理
            pthread_create(&threads[launched], NULL, pi_gemm_mn<T>, arg);
            launched++;

            // 达到一批上限
            if (launched == thread_num)
            {
                for (int t = 0; t < launched; ++t)
                    pthread_join(threads[t], NULL);
                launched = 0;
            }
        }
    }
    // 收尾：等待最后一批
    for (int t = 0; t < launched; ++t)
        pthread_join(threads[t], NULL);

    free(threads);
    free(args);
    return piSuccess;
}

piState piGemmFp64_v2(double *A, double *B, double *C,
                      double alpha, double beta,
                      size_t m, size_t k, size_t n)
{
    return piGemmV2Impl<double>(A, B, C, alpha, beta, m, k, n);
}

piState piGemmFp32_v2(float *A, float *B, float *C,
                      float alpha, float beta,
                      size_t m, size_t k, size_t n)
{
    return piGemmV2Impl<float>(A, B, C, alpha, beta, m, k, n);
}

// 兼容旧的命名，默认使用双精度
piState piGemm_v2(double *A, double *B, double *C,
                  double alpha, double beta,
                  size_t m, size_t k, size_t n)
{
    return piGemmFp64_v2(A, B, C, alpha, beta, m, k, n);
}

template <typename T>
static piState piGemmV3Impl(T *__restrict A,
                            T *__restrict B,
                            T *__restrict C,
                            T alpha, T beta,
                            size_t m, size_t k, size_t n)
{
    const size_t MB = BLOCK_SIZE, NB = BLOCK_SIZE, KB = BLOCK_SIZE;
    const int thread_num = config()->thread_num;

#pragma omp parallel num_threads(thread_num)
    {
#pragma omp for schedule(static)
        for (size_t i = 0; i < m; ++i)
        {
            T *c = C + i * n;
            if (beta == T(0))
            {
#pragma omp simd
                for (size_t j = 0; j < n; ++j)
                    c[j] = T(0);
            }
            else if (beta != T(1))
            {
#pragma omp simd
                for (size_t j = 0; j < n; ++j)
                    c[j] *= beta;
            }
        }

        for (size_t ll = 0; ll < k; ll += KB)
        {
            const size_t lmax = (ll + KB < k) ? ll + KB : k;

#pragma omp for schedule(static) collapse(2)
            for (size_t ii = 0; ii < m; ii += MB)
                for (size_t jj = 0; jj < n; jj += NB)
                {
                    const size_t imax = (ii + MB < m) ? ii + MB : m;
                    const size_t jmax = (jj + NB < n) ? jj + NB : n;

                    for (size_t i = ii; i < imax; ++i)
                    {
                        T *__restrict c = C + i * n + jj;
                        const T *__restrict arow = A + i * k + ll;

                        for (size_t l = ll; l < lmax; ++l)
                        {
                            const T aalpha = arow[l - ll] * alpha;
                            const T *__restrict brow = B + l * n + jj;

#pragma omp simd
                            for (size_t j = 0; j < (jmax - jj); ++j)
                                c[j] += aalpha * brow[j];
                        }
                    }
                }
        }
    }

    return piSuccess;
}

piState piGemmFp64_v3(double *__restrict A,
                      double *__restrict B,
                      double *__restrict C,
                      double alpha, double beta,
                      size_t m, size_t k, size_t n)
{
    return piGemmV3Impl<double>(A, B, C, alpha, beta, m, k, n);
}

piState piGemmFp32_v3(float *__restrict A,
                      float *__restrict B,
                      float *__restrict C,
                      float alpha, float beta,
                      size_t m, size_t k, size_t n)
{
    return piGemmV3Impl<float>(A, B, C, alpha, beta, m, k, n);
}

piState piGemm_v3(double *__restrict A,
                  double *__restrict B,
                  double *__restrict C,
                  double alpha, double beta,
                  size_t m, size_t k, size_t n)
{
    return piGemmFp64_v3(A, B, C, alpha, beta, m, k, n);
}
