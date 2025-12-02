#include "pi_blas.h"
#include "pi_config.h"
#include "pi_type.h"
#include "utils.h"

#include <cstddef>
#include <cstdlib>
#include <pthread.h>

typedef struct
{
    double *__restrict y;
    const double *__restrict x;
    // 一个完整一行最多占用一个线程
    const double *__restrict values;
    // 非0元素的对应的列索引数组
    const size_t *__restrict col_idx;

    // 用于全局下区分y的纬度索引
    size_t row_begin;
    size_t row_end;
    // 直接传完整的row_ptr
    const size_t *__restrict row_ptr;

} pi_spmv_worker_arg;

static void *pi_spmv_worker(void *arg_)
{
    pi_spmv_worker_arg *arg = (pi_spmv_worker_arg *)arg_;
    double *__restrict y = arg->y;
    const double *__restrict x = arg->x;
    const double *__restrict values = arg->values;
    const size_t *__restrict col_idx = arg->col_idx;
    const size_t *__restrict row_ptr = arg->row_ptr;
    const size_t row_begin = arg->row_begin;
    const size_t row_end = arg->row_end;

    for (size_t i = row_begin; i < row_end; i++)
    {
        double acc = 0.0;
        for (size_t j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            size_t cur_col_index = col_idx[j];
            // 由于是按行分线程，y的每个纬度的累加是独立的，不需要考虑写入冲突问题
            acc += x[cur_col_index] * values[j];
        }
        y[i] = acc;
    }
    return NULL;
}

piState piSpMV(const pi_csr *__restrict A, double *__restrict x, double *__restrict y)
{
    const size_t n_rows = A->n_rows;
    const size_t nnz = A->nnz;
    size_t *row_ptr = A->row_ptr;
    size_t *col_idx = A->col_idx;
    double *values = A->values;

    if (n_rows == 0)
        return piSuccess;

    // 小规模矩阵/单线程处理
    int thread_num = MIN(config()->thread_num, (int)n_rows);
    if (thread_num <= 1 || nnz < 1024)
    {
        for (size_t i = 0; i < n_rows; ++i)
        {
            double acc = 0.0;
            for (size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
                acc += x[col_idx[j]] * values[j];
            y[i] = acc;
        }
        return piSuccess;
    }

    // 根据线程数算怎么切分任务,cuts是边界索引
    size_t *cuts = (size_t *)malloc((size_t)(thread_num + 1) * sizeof(size_t));
    if (!cuts)
        return piErrAlloc;
    cuts[0] = 0;

    // 启发式思想，算出每个工作线程理想的处理的非0元素个数
    const size_t target = nnz / (size_t)thread_num + 1;
    size_t acc = 0;
    int cur = 1;
    for (size_t r = 0; r < n_rows && cur < thread_num; ++r)
    {
        acc += (row_ptr[r + 1] - row_ptr[r]);
        if (acc >= target)
        {
            cuts[cur++] = r + 1;
            acc = 0;
        }
    }
    // 若没切够，至少保证切点递增
    while (cur < thread_num)
        cuts[cur] = cuts[cur - 1] + 1, ++cur;
    cuts[thread_num] = n_rows;

    // 创建线程，准备参数
    pthread_t *th = (pthread_t *)malloc((size_t)thread_num * sizeof(pthread_t));
    pi_spmv_worker_arg *args = (pi_spmv_worker_arg *)malloc((size_t)thread_num * sizeof(pi_spmv_worker_arg));
    if (!th || !args)
    {
        free(cuts);
        free(th);
        free(args);
        return piFailure;
    }

    for (int t = 0; t < thread_num; ++t)
    {
        args[t].y = y;
        args[t].x = x;
        args[t].values = values;
        args[t].col_idx = col_idx;
        args[t].row_ptr = row_ptr;
        args[t].row_begin = cuts[t];
        args[t].row_end = cuts[t + 1];

        pthread_create(&th[t], NULL, pi_spmv_worker, &args[t]);
    }
    // 等待线程
    for (int t = 0; t < thread_num; ++t)
        pthread_join(th[t], NULL);

    free(cuts);
    free(th);
    free(args);

    return piSuccess;
}
