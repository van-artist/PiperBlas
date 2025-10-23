#include <string.h>
#include <stddef.h>
#include "utils.h"
#include "pi_blas.h"
#include "pi_type.h"

piState piSpMV(const pi_csr *A, double *x, double *y)
{
    // m是A_column_id和A_weight的数组长度
    // n是A_row_pointer的数组长度,同时也是A的行数+1
    // l_y是y的向量维度
    // l_x是x的向量维度

    size_t m = A->nnz;
    size_t n = A->n_rows + 1;
    size_t l_y = A->n_cols;
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