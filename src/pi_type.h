#pragma once
#include <stddef.h>
enum piState
{
    piSuccess = 0
};
typedef enum piState piState;

typedef struct
{
    size_t n_rows;
    size_t n_cols;
    size_t nnz;      // 非零元素个数
    size_t *row_ptr; // length n_rows + 1
    size_t *col_idx;
    double *values;
} CSR;