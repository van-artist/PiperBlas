#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "pi_type.h"
#include "utils.h"
#include "util.h"

piState csr_create(size_t n_rows, size_t n_cols, size_t nnz, pi_csr *dist)
{
    pi_csr *A = (pi_csr *)malloc(sizeof(pi_csr));
    if (!A)
        return piFailure;

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
        return piFailure;
    }
    dist = A;
    return piFailure;
}

piState csr_from_bin(const char *src_file, pi_csr *dist)
{
    if (!src_file || !dist)
        return piErrBadHeader;

    FILE *f = fopen(src_file, "rb");
    if (!f)
    {
        fprintf(stderr, "打开失败: %s (%s)\n", src_file, strerror(errno));
        return piErrOpenFile;
    }

    // 读取头部
    int32_t nrows_s32 = 0, ncols_s32 = 0;
    uint32_t nnz_u32 = 0;

    if (fread(&nrows_s32, sizeof(nrows_s32), 1, f) != 1 ||
        fread(&ncols_s32, sizeof(ncols_s32), 1, f) != 1 ||
        fread(&nnz_u32, sizeof(nnz_u32), 1, f) != 1)
    {
        fprintf(stderr, "读取头部失败（IO）。\n");
        fclose(f);
        return piErrIO;
    }

    if (nrows_s32 <= 0 || ncols_s32 <= 0)
    {
        fprintf(stderr, "非法维度: nrows=%d, ncols=%d\n", nrows_s32, ncols_s32);
        fclose(f);
        return piErrBadHeader;
    }

    const size_t n_rows = (size_t)nrows_s32;
    const size_t n_cols = (size_t)ncols_s32;
    const size_t nnz = (size_t)nnz_u32;

    // 内存分配
    size_t *row_ptr = (size_t *)malloc((n_rows + 1) * sizeof(size_t));
    size_t *col_idx = (size_t *)malloc(nnz * sizeof(size_t));
    double *values = (double *)malloc(nnz * sizeof(double));

    if (!row_ptr || !col_idx || !values)
    {
        fprintf(stderr, "内存分配失败。\n");
        pi_free((void **)&row_ptr);
        pi_free((void **)&col_idx);
        pi_free((void **)&values);
        fclose(f);
        return piErrAlloc;
    }

    // 读取 rows/cols 为 uint32_t 临时缓冲，再拓宽到 size_t
    uint32_t *tmp_rows = (uint32_t *)malloc((n_rows + 1) * sizeof(uint32_t));
    uint32_t *tmp_cols = (uint32_t *)malloc(nnz * sizeof(uint32_t));
    if (!tmp_rows || !tmp_cols)
    {
        fprintf(stderr, "临时缓冲分配失败。\n");
        pi_free((void **)&tmp_rows);
        pi_free((void **)&tmp_cols);
        pi_free((void **)&row_ptr);
        pi_free((void **)&col_idx);
        pi_free((void **)&values);
        fclose(f);
        return piErrAlloc;
    }

    // 读取三个数组段
    if (fread(tmp_rows, sizeof(uint32_t), (n_rows + 1), f) != (n_rows + 1))
    {
        fprintf(stderr, "读取 row_ptr 失败（IO）。\n");
        pi_free((void **)&tmp_rows);
        pi_free((void **)&tmp_cols);
        pi_free((void **)&row_ptr);
        pi_free((void **)&col_idx);
        pi_free((void **)&values);
        fclose(f);
        return piErrIO;
    }
    if (fread(tmp_cols, sizeof(uint32_t), nnz, f) != nnz)
    {
        fprintf(stderr, "读取 col_idx 失败（IO）。\n");
        pi_free((void **)&tmp_rows);
        pi_free((void **)&tmp_cols);
        pi_free((void **)&row_ptr);
        pi_free((void **)&col_idx);
        pi_free((void **)&values);
        fclose(f);
        return piErrIO;
    }
    if (fread(values, sizeof(double), nnz, f) != nnz)
    {
        fprintf(stderr, "读取 values 失败（IO）。\n");
        pi_free((void **)&tmp_rows);
        pi_free((void **)&tmp_cols);
        pi_free((void **)&row_ptr);
        pi_free((void **)&col_idx);
        pi_free((void **)&values);
        fclose(f);
        return piErrIO;
    }
    fclose(f);

    // 拓宽
    if (tmp_rows[0] != 0U)
    {
        fprintf(stderr, "非法 CSR：row_ptr[0] 应为 0，实际=%u\n", tmp_rows[0]);
        goto csr_invalid;
    }
    for (size_t i = 0; i < n_rows; ++i)
    {
        if (tmp_rows[i] > tmp_rows[i + 1])
        {
            fprintf(stderr, "非法 CSR：row_ptr 非递增，在 i=%zu 处出现 %u > %u\n",
                    i, tmp_rows[i], tmp_rows[i + 1]);
            goto csr_invalid;
        }
    }
    if (tmp_rows[n_rows] != nnz_u32)
    {
        fprintf(stderr, "非法 CSR：row_ptr[n_rows](=%u) != nnz(=%u)\n",
                tmp_rows[n_rows], nnz_u32);
        goto csr_invalid;
    }

    for (size_t i = 0; i < n_rows + 1; ++i)
        row_ptr[i] = (size_t)tmp_rows[i];
    for (size_t k = 0; k < nnz; ++k)
    {
        if (tmp_cols[k] >= n_cols)
        {
            fprintf(stderr, "非法 CSR：col_idx[%zu]=%u 超界 (n_cols=%zu)\n",
                    k, tmp_cols[k], n_cols);
            goto csr_invalid;
        }
        col_idx[k] = (size_t)tmp_cols[k];
    }

    dist->n_rows = n_rows;
    dist->n_cols = n_cols;
    dist->nnz = nnz;
    dist->row_ptr = row_ptr;
    dist->col_idx = col_idx;
    dist->values = values;

    pi_free((void **)&tmp_rows);
    pi_free((void **)&tmp_cols);
    return piSuccess;

csr_invalid:
    pi_free((void **)&tmp_rows);
    pi_free((void **)&tmp_cols);
    pi_free((void **)&row_ptr);
    pi_free((void **)&col_idx);
    pi_free((void **)&values);
    return piErrCSRInvalid;
}

void csr_destroy(pi_csr *A)
{
    if (!A)
        return;
    free(A->row_ptr);
    free(A->col_idx);
    free(A->values);
}

void csr_print(const pi_csr *A)
{
    if (!A)
        return;
    printf("pi_csr Matrix: %zu x %zu, nnz = %zu\n", A->n_rows, A->n_cols, A->nnz);
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
