#include <utils.hpp>
#include <cstddef>

piState piGemm(double *A, double *B, double *C, double alpha, double beta, size_t m, size_t k, size_t n);
piState piGeMV(double *A, double *B, double *C, double alpha, double beta, size_t m, size_t k);
piState piSpMV(size_t *A_column_id, size_t *A_row_pointer, double *A_weight, double *x, double *y, size_t m, size_t n, size_t l_x, size_t l_y);