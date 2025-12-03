#pragma once
#include <cuda_runtime.h>
#include "pi_type.h"

piState axpy(int n, float alpha, const float *__restrict__ x, float *__restrict__ y);
piState gemm_fp32(float *A, float *B, float *C, float alpha, float beta, int m, int k, int n);
piState gemm_fp64(double *A, double *B, double *C, double alpha, double beta, int m, int k, int n);
