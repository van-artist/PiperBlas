#pragma once
#include "pi_type.hpp"

typedef struct
{
    int n_rows;
    int n_cols;
    int nnz;
    int *row_ptr;
    int *col_idx;
    double *values;
} pi_csr;

piState csr_create(int n_rows, int n_cols, int nnz, pi_csr *dist);
void csr_destroy(pi_csr *A);
void csr_print(const pi_csr *A);
piState csr_from_bin(const char *src_file, pi_csr *dist);
