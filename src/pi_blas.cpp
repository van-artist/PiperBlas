#include "utils.hpp"
#include <cstddef>
#include <cstring>

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

    return piSucsess;
}

piState piGeMV(double *A, double *B, double *C, double alpha, double beta, size_t m, size_t k)
{
    return piSucsess;
}

piState piSpMV(size_t *A_column_id, size_t *A_row_pointer, double *A_weight, double *x, double *y, size_t m, size_t n, size_t l_x, size_t l_y)
{
    // m是A_column_id和A_weight的数组长度
    // n是A_row_pointer的数组长度,同时也是A的行数+1
    // l_y是y的向量维度
    // l_x是x的向量维度
    memset(y, 0, l_y * sizeof(double));
    for (size_t i = 0; i < n - 1; i++)
    {
        size_t cur_row_len = A_row_pointer[i + 1] - A_row_pointer[i];
        for (size_t j = A_row_pointer[i]; j < A_row_pointer[i + 1]; j++)
        {
            size_t cur_col_index = A_column_id[j];
            y[i] += x[cur_col_index] * A_weight[j];
        }
    }

    return piSucsess;
}