#include "pi_type.h"
#include "utils.h"
#include "pi_csr.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <limits>

piState csr_create(int n_rows, int n_cols, int nnz, pi_csr *dist)
{
    if (!dist)
        return piErrBadHeader;
    if (n_rows < 0 || n_cols < 0 || nnz < 0)
        return piErrBadHeader;

    auto row_ptr = std::unique_ptr<int[], decltype(&std::free)>(
        static_cast<int *>(std::calloc((size_t)n_rows + 1, sizeof(int))), &std::free);
    auto col_idx = std::unique_ptr<int[], decltype(&std::free)>(
        static_cast<int *>(std::calloc((size_t)nnz, sizeof(int))), &std::free);
    auto values = std::unique_ptr<double[], decltype(&std::free)>(
        static_cast<double *>(std::calloc((size_t)nnz, sizeof(double))), &std::free);

    if (!row_ptr || !col_idx || !values)
        return piErrAlloc;

    dist->n_rows = n_rows;
    dist->n_cols = n_cols;
    dist->nnz = nnz;
    dist->row_ptr = row_ptr.release();
    dist->col_idx = col_idx.release();
    dist->values = values.release();
    return piSuccess;
}

piState csr_from_bin(const char *src_file, pi_csr *dist)
{
    if (!src_file || !dist)
        return piErrBadHeader;

    FILE *f = std::fopen(src_file, "rb");
    if (!f)
    {
        std::fprintf(stderr, "打开失败: %s (%s)\n", src_file, std::strerror(errno));
        return piErrOpenFile;
    }

    int32_t nrows_s32 = 0, ncols_s32 = 0;
    uint32_t nnz_u32 = 0;

    if (std::fread(&nrows_s32, sizeof(nrows_s32), 1, f) != 1 ||
        std::fread(&ncols_s32, sizeof(ncols_s32), 1, f) != 1 ||
        std::fread(&nnz_u32, sizeof(nnz_u32), 1, f) != 1)
    {
        std::fprintf(stderr, "读取头部失败（IO）。\n");
        std::fclose(f);
        return piErrIO;
    }

    if (nrows_s32 <= 0 || ncols_s32 <= 0)
    {
        std::fprintf(stderr, "非法维度: nrows=%d, ncols=%d\n", nrows_s32, ncols_s32);
        std::fclose(f);
        return piErrBadHeader;
    }

    if (nrows_s32 > std::numeric_limits<int>::max() ||
        ncols_s32 > std::numeric_limits<int>::max() ||
        nnz_u32 > (uint32_t)std::numeric_limits<int>::max())
    {
        std::fprintf(stderr, "维度/nnz 超出 int 范围: nrows=%d ncols=%d nnz=%u\n",
                     nrows_s32, ncols_s32, nnz_u32);
        std::fclose(f);
        return piErrBadHeader;
    }

    const int n_rows = (int)nrows_s32;
    const int n_cols = (int)ncols_s32;
    const int nnz = (int)nnz_u32;

    auto row_ptr = std::unique_ptr<int[], decltype(&std::free)>(
        static_cast<int *>(std::malloc(((size_t)n_rows + 1) * sizeof(int))), &std::free);
    auto col_idx = std::unique_ptr<int[], decltype(&std::free)>(
        static_cast<int *>(std::malloc((size_t)nnz * sizeof(int))), &std::free);
    auto values = std::unique_ptr<double[], decltype(&std::free)>(
        static_cast<double *>(std::malloc((size_t)nnz * sizeof(double))), &std::free);

    if (!row_ptr || !col_idx || !values)
    {
        std::fprintf(stderr, "内存分配失败。\n");
        std::fclose(f);
        return piErrAlloc;
    }

    auto tmp_rows = std::unique_ptr<uint32_t[], decltype(&std::free)>(
        static_cast<uint32_t *>(std::malloc(((size_t)n_rows + 1) * sizeof(uint32_t))), &std::free);
    auto tmp_cols = std::unique_ptr<uint32_t[], decltype(&std::free)>(
        static_cast<uint32_t *>(std::malloc((size_t)nnz * sizeof(uint32_t))), &std::free);
    if (!tmp_rows || !tmp_cols)
    {
        std::fprintf(stderr, "临时缓冲分配失败。\n");
        std::fclose(f);
        return piErrAlloc;
    }

    if (std::fread(tmp_rows.get(), sizeof(uint32_t), (size_t)n_rows + 1, f) != (size_t)n_rows + 1)
    {
        std::fprintf(stderr, "读取 row_ptr 失败（IO）。\n");
        std::fclose(f);
        return piErrIO;
    }
    if (std::fread(tmp_cols.get(), sizeof(uint32_t), (size_t)nnz, f) != (size_t)nnz)
    {
        std::fprintf(stderr, "读取 col_idx 失败（IO）。\n");
        std::fclose(f);
        return piErrIO;
    }
    if (std::fread(values.get(), sizeof(double), (size_t)nnz, f) != (size_t)nnz)
    {
        std::fprintf(stderr, "读取 values 失败（IO）。\n");
        std::fclose(f);
        return piErrIO;
    }
    std::fclose(f);

    if (tmp_rows[0] != 0U)
    {
        std::fprintf(stderr, "非法 CSR：row_ptr[0] 应为 0，实际=%u\n", tmp_rows[0]);
        return piErrCSRInvalid;
    }
    for (int i = 0; i < n_rows; ++i)
    {
        if (tmp_rows[i] > tmp_rows[i + 1])
        {
            std::fprintf(stderr, "非法 CSR：row_ptr 非递增，在 i=%d 处出现 %u > %u\n",
                         i, tmp_rows[i], tmp_rows[i + 1]);
            return piErrCSRInvalid;
        }
        if (tmp_rows[i + 1] > nnz_u32)
        {
            std::fprintf(stderr, "非法 CSR：row_ptr[%d]=%u 超出 nnz=%u\n",
                         i + 1, tmp_rows[i + 1], nnz_u32);
            return piErrCSRInvalid;
        }
    }
    if (tmp_rows[n_rows] != nnz_u32)
    {
        std::fprintf(stderr, "非法 CSR：row_ptr[n_rows](=%u) != nnz(=%u)\n",
                     tmp_rows[n_rows], nnz_u32);
        return piErrCSRInvalid;
    }

    for (int i = 0; i < n_rows + 1; ++i)
        row_ptr[i] = (int)tmp_rows[i];

    for (int k = 0; k < nnz; ++k)
    {
        if (tmp_cols[k] >= (uint32_t)n_cols)
        {
            std::fprintf(stderr, "非法 CSR：col_idx[%d]=%u 超界 (n_cols=%d)\n",
                         k, tmp_cols[k], n_cols);
            return piErrCSRInvalid;
        }
        col_idx[k] = (int)tmp_cols[k];
    }

    dist->n_rows = n_rows;
    dist->n_cols = n_cols;
    dist->nnz = nnz;
    dist->row_ptr = row_ptr.release();
    dist->col_idx = col_idx.release();
    dist->values = values.release();

    return piSuccess;
}

void csr_destroy(pi_csr *A)
{
    if (!A)
        return;
    std::free(A->row_ptr);
    std::free(A->col_idx);
    std::free(A->values);
    A->row_ptr = nullptr;
    A->col_idx = nullptr;
    A->values = nullptr;
    A->n_rows = 0;
    A->n_cols = 0;
    A->nnz = 0;
}

void csr_print(const pi_csr *A)
{
    if (!A)
        return;
    std::printf("pi_csr Matrix: %d x %d, nnz = %d\n", A->n_rows, A->n_cols, A->nnz);
    std::printf("row_ptr: ");
    for (int i = 0; i < A->n_rows + 1; ++i)
        std::printf("%d ", A->row_ptr[i]);
    std::printf("\ncol_idx: ");
    for (int i = 0; i < A->nnz; ++i)
        std::printf("%d ", A->col_idx[i]);
    std::printf("\nvalues:  ");
    for (int i = 0; i < A->nnz; ++i)
        std::printf("%.3f ", A->values[i]);
    std::printf("\n");
}
