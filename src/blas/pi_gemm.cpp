#include <stddef.h>
#include <stdlib.h>
#include <pthread.h>

#include "pi_blas.hpp"
#include "core/pi_config.hpp"

#define BLOCK_SIZE 128

template <typename T>
struct pi_gemm_arg
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
    pi_gemm_arg<T> *arg = (pi_gemm_arg<T> *)arg_;

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
static piState piGemmImpl(T *A, T *B, T *C,
                          T alpha, T beta,
                          size_t m, size_t k, size_t n)
{
    const int thread_num_cfg = PiConfig::instance().thread_num();
    const int thread_num = (thread_num_cfg > 0) ? thread_num_cfg : 1;

    const size_t MB = BLOCK_SIZE;
    const size_t NB = BLOCK_SIZE;
    const size_t KB = BLOCK_SIZE;

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

                pi_gemm_arg<T> arg;
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

    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)thread_num);
    pi_gemm_arg<T> *args = (pi_gemm_arg<T> *)malloc(sizeof(pi_gemm_arg<T>) * (size_t)thread_num);
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

            pi_gemm_arg<T> *arg = &args[launched];
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

            pthread_create(&threads[launched], NULL, pi_gemm_mn<T>, arg);
            launched++;

            if (launched == thread_num)
            {
                for (int t = 0; t < launched; ++t)
                    pthread_join(threads[t], NULL);
                launched = 0;
            }
        }
    }

    for (int t = 0; t < launched; ++t)
        pthread_join(threads[t], NULL);

    free(threads);
    free(args);
    return piSuccess;
}

piState pi_gemm_fp64(double *A, double *B, double *C,
                     double alpha, double beta,
                     size_t m, size_t k, size_t n)
{
    return piGemmImpl<double>(A, B, C, alpha, beta, m, k, n);
}

piState pi_gemm_fp32(float *A, float *B, float *C,
                     float alpha, float beta,
                     size_t m, size_t k, size_t n)
{
    return piGemmImpl<float>(A, B, C, alpha, beta, m, k, n);
}
