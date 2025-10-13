#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>
#include "pi_type.h"
    piState piSpMV(const CSR *A, double *x, double *y, size_t l_x, size_t l_y);
    piState piGemm(double *A, double *B, double *C, double alpha, double beta, size_t m, size_t k, size_t n);

#ifdef __cplusplus
}
#endif
