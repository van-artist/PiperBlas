#include <stdlib.h>
#include <stdio.h>
#include "pi_type.h"

CSR *csr_create(size_t n_rows, size_t n_cols, size_t nnz)
{
    CSR *A = (CSR *)malloc(sizeof(CSR));
    if (!A)
        return NULL;

    A->n_rows = n_rows;
    A->n_cols = n_cols;
    A->nnz = nnz;

    A->row_ptr = (size_t *)calloc(n_rows + 1, sizeof(size_t));
    A->col_idx = (size_t *)calloc(nnz, sizeof(size_t));
    A->values = (double *)calloc(nnz, sizeof(double));

    if (!A->row_ptr || !A->col_idx || !A->values)
    {
        free(A->row_ptr);
        free(A->col_idx);
        free(A->values);
        free(A);
        return NULL;
    }
    return A;
}

void csr_destroy(CSR *A)
{
    if (!A)
        return;
    free(A->row_ptr);
    free(A->col_idx);
    free(A->values);
    free(A);
}

void csr_print(const CSR *A)
{
    if (!A)
        return;
    printf("CSR Matrix: %zu x %zu, nnz = %zu\n", A->n_rows, A->n_cols, A->nnz);
    printf("row_ptr: ");
    for (size_t i = 0; i < A->n_rows + 1; ++i)
        printf("%zu ", A->row_ptr[i]);
    printf("\ncol_idx: ");
    for (size_t i = 0; i < A->nnz; ++i)
        printf("%zu ", A->col_idx[i]);
    printf("\nvalues:  ");
    for (size_t i = 0; i < A->nnz; ++i)
        printf("%.3f ", A->values[i]);
    printf("\n");
}