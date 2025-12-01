#pragma once

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

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

void pi_free(void **p);
void print_cuda_important_attrs(int device);