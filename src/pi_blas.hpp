#include <utils.hpp>
#include <cstddef>

piState piGemm(double *A, double *B, double *C, double alpha, double beta, size_t m, size_t k, size_t n);
piState piGeMV(double *A, double *B, double *C, double alpha, double beta, size_t m, size_t k);
piState piSpMV(size_t *column_id, size_t *row_pointer, double *value, size_t m, size_t n);