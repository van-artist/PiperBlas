#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "utils.h"
#include "pi_blas.h"
#include "pi_type.h"
#include "pi_config.h"

typedef struct
{
    double *A;
    double *B;
    double *C;
    double alpha;
    double beta;
    size_t m;
    size_t k;
    size_t n;
} pi_gemm_arg;

typedef struct
{
    piState state;
} pi_gemm_retval;

// 按M纬切分下单个线程的处理
static void *pi_gemm_m(void *arg_)
{
    pi_gemm_arg *arg = (pi_gemm_arg *)arg_;
    for (size_t i = 0; i < arg->m; i++)
    {
        for (size_t j = 0; j < arg->n; j++)
        {
            size_t offset_c = i * arg->n + j;
            double sum = 0.0;
            for (size_t l = 0; l < arg->k; l++)
            {
                size_t offset_a = i * arg->k + l;
                size_t offset_b = l * arg->n + j;
                sum += arg->A[offset_a] * arg->B[offset_b];
            }
            arg->C[offset_c] = arg->alpha * sum + arg->beta * arg->C[offset_c];
        }
    }
    pi_gemm_retval *retval = (pi_gemm_retval *)malloc(sizeof(pi_gemm_retval));
    if (!retval)
        return NULL;
    retval->state = piSuccess;
    return (void *)retval;
}

piState piGemm(double *A, double *B, double *C, double alpha, double beta, size_t m, size_t k, size_t n)
{
    // A是m行k列
    // B是k行n列
    // C是m行n列
    // 按照M纬度划分任务
    int thread_num = config()->thread_num;
    // M特别小的时候，直接单线程计算，这个后续可以丰富调度策略
    if (thread_num <= 1 || m < (size_t)thread_num)
    {
        pi_gemm_arg arg;
        arg.A = A;
        arg.B = B;
        arg.C = C;
        arg.alpha = alpha;
        arg.beta = beta;
        arg.m = m;
        arg.k = k;
        arg.n = n;
        pi_gemm_m(&arg);
        return piSuccess;
    }

    // 准备参数
    pi_gemm_arg *args = malloc(sizeof(pi_gemm_arg) * thread_num);
    int single_m = m / thread_num;
    int last_m = m % thread_num;
    int has_last = last_m;
    for (int i = 0; i < thread_num; i++)
    {
        args[i].alpha = alpha;
        args[i].B = B;
        args[i].k = k;
        args[i].n = n;
        args[i].m = single_m;
        if (has_last && i == thread_num - 1)
        {
            args[i].m = single_m + last_m;
        }
        size_t offset_a = (size_t)i * (size_t)single_m * k;
        size_t offset_c = (size_t)i * (size_t)single_m * n;
        args[i].A = A + offset_a;
        args[i].C = C + offset_c;
    }
    // 创建线程
    pthread_t *threads = malloc(sizeof(pthread_t) * thread_num);

    for (int i = 0; i < thread_num; i++)
    {
        pthread_create(&threads[i], NULL, pi_gemm_m, &args[i]);
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