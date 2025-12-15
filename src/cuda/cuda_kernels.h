#pragma once
#include <cuda_runtime.h>
#include "pi_type.h"
#include "pi_csr.h"

piState axpy(int n, float alpha, const float *__restrict__ x, float *__restrict__ y);
piState piCudaGemmFp32(float *__restrict__ A,
                       float *__restrict__ B,
                       float *__restrict__ C,
                       float alpha,
                       float beta,
                       int M, int K, int N);
piState piCudaGemmFp64(double *__restrict__ A,
                       double *__restrict__ B,
                       double *__restrict__ C,
                       double alpha,
                       double beta,
                       int M, int K, int N);
piState piCudaGemmFp32_v2(float *__restrict__ A,
                          float *__restrict__ B,
                          float *__restrict__ C,
                          float alpha,
                          float beta,
                          int M, int K, int N);
piState piCudaGemmFp64_v2(double *__restrict__ A,
                          double *__restrict__ B,
                          double *__restrict__ C,
                          double alpha,
                          double beta,
                          int M, int K, int N);
piState piCudaGemmFp32_v3(float *__restrict__ A,
                          float *__restrict__ B,
                          float *__restrict__ C,
                          float alpha,
                          float beta,
                          int M, int K, int N);
piState piCudaGemmFp32_v4(float *__restrict__ A,
                          float *__restrict__ B,
                          float *__restrict__ C,
                          float alpha,
                          float beta,
                          int M, int K, int N);

piState pi_cuda_spmv_fp32(const pi_csr *__restrict A, double *__restrict x, double *__restrict y);
piState pi_cuda_spmv_fp64(const pi_csr *__restrict A, double *__restrict x, double *__restrict y);