#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                            \
    do                                                              \
    {                                                               \
        cudaError_t err__ = (call);                                 \
        if (err__ != cudaSuccess)                                   \
        {                                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",           \
                         __FILE__, __LINE__, cudaGetErrorString(err__)); \
            std::exit(EXIT_FAILURE);                                \
        }                                                           \
    } while (0)

#define CHECK_CUBLAS(call)                                \
    do                                                    \
    {                                                     \
        cublasStatus_t st__ = (call);                     \
        if (st__ != CUBLAS_STATUS_SUCCESS)                \
        {                                                 \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", \
                         (int)st__, __FILE__, __LINE__);  \
            std::exit(1);                                 \
        }                                                 \
    } while (0)

void print_cuda_info(int device);

