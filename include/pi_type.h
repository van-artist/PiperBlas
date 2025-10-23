#pragma once
#include <stddef.h>
enum piState
{
    piSuccess = 0,
    piErrOpenFile = 1,
    piErrBadHeader = 2,
    piErrAlloc = 3,
    piErrIO = 4,
    piErrCSRInvalid = 5,
    piDataInvalid = 6,
    piFailure = -1
};
typedef enum piState piState;

typedef struct
{
    size_t n_rows;
    size_t n_cols;
    size_t nnz;      // 非零元素个数
    size_t *row_ptr; // 长度为n_rows + 1
    size_t *col_idx;
    double *values;
} pi_csr;