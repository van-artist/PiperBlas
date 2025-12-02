#pragma once
#include <cuda_runtime.h>
#include "pi_type.h"

__global__ void axpy_kernel(int n, float alpha, const float *x, float *y);
__global__ void gemm_kernel(int m, int k, int n, const float *A, const float *B, float *C, const float alpha, const float beta);
piState axpy(int n, float alpha, const float *__restrict__ x, float *__restrict__ y);
piState gemm(int m, int k, int n, const float *A, const float *B, float *C, const float alpha, const float beta);