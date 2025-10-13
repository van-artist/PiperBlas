#include <string.h>
#include <stddef.h>
#include "utils.h"
#include "pi_blas.h"
#include "pi_type.h"

piState piGemm(double *A, double *B, double *C, double alpha, double beta, size_t m, size_t k, size_t n)
{
    // A是m行k列
    // B是k行n列
    // C是m行n列
    for (size_t i = 0; i < m; i++)
    {

        for (size_t j = 0; j < n; j++)
        {
            size_t offset_c = i * n + j;

            double sum = 0;
            for (size_t l = 0; l < k; l++)
            {
                size_t offset_a = i * k + l;
                size_t offset_b = l * n + j;
                sum += A[offset_a] * B[offset_b];
            }
            C[offset_c] = alpha * sum + beta * C[offset_c];
        }
    }

    return piSuccess;
}

piState piGeMV(double *A, double *B, double *C, double alpha, double beta, size_t m, size_t k)
{
    return piSuccess;
}

piState piSpMV(const CSR *A, double *x, double *y,
               size_t l_x, size_t l_y)
{
    // m是A_column_id和A_weight的数组长度
    // n是A_row_pointer的数组长度,同时也是A的行数+1
    // l_y是y的向量维度
    // l_x是x的向量维度

    size_t m = A->nnz;
    size_t n = A->n_rows + 1;

    memset(y, 0, l_y * sizeof(double));
    for (size_t i = 0; i < n - 1; i++)
    {
        size_t cur_row_len = A->row_ptr[i + 1] - A->row_ptr[i];
        for (size_t j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++)
        {
            size_t cur_col_index = A->col_idx[j];
            y[i] += x[cur_col_index] * A->values[j];
        }
    }

    return piSuccess;
}