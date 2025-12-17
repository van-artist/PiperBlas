#pragma once

#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#if defined(PIPER_HAVE_CUDA) && PIPER_HAVE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
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
#else
#define CHECK_CUDA(call)                                \
    do                                                  \
    {                                                   \
        (void)(call);                                   \
        fprintf(stderr, "CUDA support is disabled.\n"); \
        std::exit(EXIT_FAILURE);                        \
    } while (0)

#define CHECK_CUBLAS(call)                                \
    do                                                    \
    {                                                     \
        (void)(call);                                     \
        fprintf(stderr, "cuBLAS support is disabled.\n"); \
        std::exit(EXIT_FAILURE);                          \
    } while (0)
#endif

void pi_free(void **p);
double wall_now();

#if defined(PIPER_HAVE_CUDA) && PIPER_HAVE_CUDA
void print_cuda_info(int device);
#endif
