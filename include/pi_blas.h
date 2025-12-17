#pragma once
#include <stddef.h>
#include "pi_type.h"
#include "pi_csr.h"
piState piSpMV(const pi_csr *A, double *x, double *y);
piState piGemmFp32(float *A, float *B, float *C,
                   float alpha, float beta,
                   size_t m, size_t k, size_t n);
piState piGemmFp64(double *A, double *B, double *C,
                   double alpha, double beta,
                   size_t m, size_t k, size_t n);
