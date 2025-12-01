#include <cuda_runtime.h>
#include "cuda/axpy_kernel.h"
#include <iostream>

template <typename Iterable>
void print_container(const Iterable &cont)
{
    std::cout << "{ ";
    for (const auto &x : cont)
        std::cout << x << " ";
    std::cout << "}\n";
}
template <typename T>
void print_array(const T *data, int n)
{
    std::cout << "{ ";
    for (int i = 0; i < n; ++i)
    {
        std::cout << data[i] << " ";
    }
    std::cout << "}\n";
}

constexpr int BLOCK_DIM = 256;

int main()
{
    int n = 333;
    int *A_device, *B_device, *C_device;
    int *A_host, *B_host, *C_host;
    A_host = (int *)malloc(n * sizeof(int));
    B_host = (int *)malloc(n * sizeof(int));
    C_host = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        A_host[i] = i;
        B_host[i] = i;
        C_host[i] = 0;
    }

    cudaMalloc(&A_device, n * sizeof(int));
    cudaMalloc(&B_device, n * sizeof(int));
    cudaMalloc(&C_device, n * sizeof(int));
    cudaMemcpy(A_device, A_host, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, n * sizeof(int), cudaMemcpyHostToDevice);
    int grid_dim = (n + BLOCK_DIM + 1) / BLOCK_DIM;
    axpy_kernel<<<grid_dim, BLOCK_DIM>>>(n, A_device, B_device, C_device);
    cudaMemcpy(C_host, C_device, n * sizeof(int), cudaMemcpyDeviceToHost);
    print_array(C_host, n);
}