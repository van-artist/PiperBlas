#pragma once
#include "pi_type.h"

typedef struct
{
    size_t n_rows;
    size_t n_cols;
    size_t nnz;      // 非零元素个数
    size_t *row_ptr; // 长度为n_rows + 1
    size_t *col_idx;
    double *values;
} pi_csr;
piState csr_create(size_t n_rows, size_t n_cols, size_t nnz, pi_csr *dist);
void csr_destroy(pi_csr *A);
void csr_print(const pi_csr *A);
piState csr_from_bin(const char *src_file, pi_csr *dist);