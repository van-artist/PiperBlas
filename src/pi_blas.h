#pragma once
#include <stddef.h>
#include "pi_type.h"
piState piSpMV(const pi_csr *A, double *x, double *y);
// fp32/fp64 GEMM; 保留旧接口以兼容历史调用
piState piGemmFp32(float *A, float *B, float *C, float alpha, float beta, size_t m, size_t k, size_t n);
piState piGemmFp32_v2(float *A, float *B, float *C,
                      float alpha, float beta,
                      size_t m, size_t k, size_t n);
piState piGemmFp32_v3(float *A,
                      float *B,
                      float *C,
                      float alpha, float beta,
                      size_t m, size_t k, size_t n);

piState piGemmFp64(double *A, double *B, double *C, double alpha, double beta, size_t m, size_t k, size_t n);
piState piGemmFp64_v2(double *A, double *B, double *C,
                      double alpha, double beta,
                      size_t m, size_t k, size_t n);
piState piGemmFp64_v3(double *A,
                      double *B,
                      double *C,
                      double alpha, double beta,
                      size_t m, size_t k, size_t n);

piState piGemm(double *A, double *B, double *C, double alpha, double beta, size_t m, size_t k, size_t n);
piState piGemm_v2(double *A, double *B, double *C,
                  double alpha, double beta,
                  size_t m, size_t k, size_t n);
piState piGemm_v3(double *A,
                  double *B,
                  double *C,
                  double alpha, double beta,
                  size_t m, size_t k, size_t n);
