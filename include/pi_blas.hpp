#pragma once
#include <stddef.h>
#include "pi_type.hpp"
#include "pi_csr.hpp"
piState pi_spmv(const pi_csr *A, double *x, double *y);
piState pi_gemm_fp32(float *A, float *B, float *C,
                     float alpha, float beta,
                     size_t m, size_t k, size_t n);
piState pi_gemm_fp64(double *A, double *B, double *C,
                     double alpha, double beta,
                     size_t m, size_t k, size_t n);
piState pi_summa_fp64(double *A, double *B, double *C,
                      double alpha, double beta,
                      int M, int K, int N, int pr, int pc, int mb, int nb);
piState pi_summa_fp32(float *A, float *B, float *C,
                      float alpha, float beta,
                      int m, int k, int n, int pr, int pc, int mb, int nb);
