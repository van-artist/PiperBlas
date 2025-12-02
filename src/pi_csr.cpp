#include "pi_type.h"
#include "utils.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

piState csr_create(size_t n_rows, size_t n_cols, size_t nnz, pi_csr *dist)
{
    if (!dist)
        return piErrBadHeader;

    auto row_ptr = std::unique_ptr<size_t[], decltype(&std::free)>(
        static_cast<size_t *>(std::calloc(n_rows + 1, sizeof(size_t))), &std::free);
    auto col_idx = std::unique_ptr<size_t[], decltype(&std::free)>(
        static_cast<size_t *>(std::calloc(nnz, sizeof(size_t))), &std::free);
    auto values = std::unique_ptr<double[], decltype(&std::free)>(
        static_cast<double *>(std::calloc(nnz, sizeof(double))), &std::free);

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

    // 读取头部
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

    const size_t n_rows = (size_t)nrows_s32;
    const size_t n_cols = (size_t)ncols_s32;
    const size_t nnz = (size_t)nnz_u32;

    // 内存分配
    auto row_ptr = std::unique_ptr<size_t[], decltype(&std::free)>(
        static_cast<size_t *>(std::malloc((n_rows + 1) * sizeof(size_t))), &std::free);
    auto col_idx = std::unique_ptr<size_t[], decltype(&std::free)>(
        static_cast<size_t *>(std::malloc(nnz * sizeof(size_t))), &std::free);
    auto values = std::unique_ptr<double[], decltype(&std::free)>(
        static_cast<double *>(std::malloc(nnz * sizeof(double))), &std::free);

    if (!row_ptr || !col_idx || !values)
    {
        std::fprintf(stderr, "内存分配失败。\n");
        std::fclose(f);
        return piErrAlloc;
    }

    // 读取 rows/cols 为 uint32_t 临时缓冲，再拓宽到 size_t
    auto tmp_rows = std::unique_ptr<uint32_t[], decltype(&std::free)>(
        static_cast<uint32_t *>(std::malloc((n_rows + 1) * sizeof(uint32_t))), &std::free);
    auto tmp_cols = std::unique_ptr<uint32_t[], decltype(&std::free)>(
        static_cast<uint32_t *>(std::malloc(nnz * sizeof(uint32_t))), &std::free);
    if (!tmp_rows || !tmp_cols)
    {
        std::fprintf(stderr, "临时缓冲分配失败。\n");
        std::fclose(f);
        return piErrAlloc;
    }

    // 读取三个数组段
    if (std::fread(tmp_rows.get(), sizeof(uint32_t), (n_rows + 1), f) != (n_rows + 1))
    {
        std::fprintf(stderr, "读取 row_ptr 失败（IO）。\n");
        std::fclose(f);
        return piErrIO;
    }
    if (std::fread(tmp_cols.get(), sizeof(uint32_t), nnz, f) != nnz)
    {
        std::fprintf(stderr, "读取 col_idx 失败（IO）。\n");
        std::fclose(f);
        return piErrIO;
    }
    if (std::fread(values.get(), sizeof(double), nnz, f) != nnz)
    {
        std::fprintf(stderr, "读取 values 失败（IO）。\n");
        std::fclose(f);
        return piErrIO;
    }
    std::fclose(f);

    // 拓宽
    if (tmp_rows[0] != 0U)
    {
        std::fprintf(stderr, "非法 CSR：row_ptr[0] 应为 0，实际=%u\n", tmp_rows[0]);
        return piErrCSRInvalid;
    }
    for (size_t i = 0; i < n_rows; ++i)
    {
        if (tmp_rows[i] > tmp_rows[i + 1])
        {
            std::fprintf(stderr, "非法 CSR：row_ptr 非递增，在 i=%zu 处出现 %u > %u\n",
                         i, tmp_rows[i], tmp_rows[i + 1]);
            return piErrCSRInvalid;
        }
    }
    if (tmp_rows[n_rows] != nnz_u32)
    {
        std::fprintf(stderr, "非法 CSR：row_ptr[n_rows](=%u) != nnz(=%u)\n",
                     tmp_rows[n_rows], nnz_u32);
        return piErrCSRInvalid;
    }

    for (size_t i = 0; i < n_rows + 1; ++i)
        row_ptr[i] = static_cast<size_t>(tmp_rows[i]);
    for (size_t k = 0; k < nnz; ++k)
    {
        if (tmp_cols[k] >= n_cols)
        {
            std::fprintf(stderr, "非法 CSR：col_idx[%zu]=%u 超界 (n_cols=%zu)\n",
                         k, tmp_cols[k], n_cols);
            return piErrCSRInvalid;
        }
        col_idx[k] = static_cast<size_t>(tmp_cols[k]);
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
}

void csr_print(const pi_csr *A)
{
    if (!A)
        return;
    std::printf("pi_csr Matrix: %zu x %zu, nnz = %zu\n", A->n_rows, A->n_cols, A->nnz);
    std::printf("row_ptr: ");
    for (size_t i = 0; i < A->n_rows + 1; ++i)
        std::printf("%zu ", A->row_ptr[i]);
    std::printf("\ncol_idx: ");
    for (size_t i = 0; i < A->nnz; ++i)
        std::printf("%zu ", A->col_idx[i]);
    std::printf("\nvalues:  ");
    for (size_t i = 0; i < A->nnz; ++i)
        std::printf("%.3f ", A->values[i]);
    std::printf("\n");
}
