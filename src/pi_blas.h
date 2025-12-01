#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>
#include "pi_type.h"
    piState piSpMV(const pi_csr *A, double *x, double *y);
    piState piGemm(double *A, double *B, double *C, double alpha, double beta, size_t m, size_t k, size_t n);
    piState piGemm_v2(double *A, double *B, double *C,
                      double alpha, double beta,
                      size_t m, size_t k, size_t n);
#ifdef __cplusplus
}
#endif
