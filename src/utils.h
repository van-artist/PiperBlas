#pragma once

#include <cuda_runtime.h>
#define CHECK_CUDA(call)                                            \
    do                                                              \
    {                                                               \
        cudaError_t err__ = (call);                                 \
        if (err__ != cudaSuccess)                                   \
        {                                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n",               \
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
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", \
                    (int)st__, __FILE__, __LINE__);       \
            std::exit(1);                                 \
        }                                                 \
    } while (0)

void pi_free(void **p);
void print_cuda_important_attrs(int device);