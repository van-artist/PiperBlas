#include <iostream>
#include "core/common.hpp"

void print_cpu_info();

#if PIPER_HAVE_CUDA
#include "cuda/cuda_common.cuh"
#endif

int main()
{
    std::cout << "Hello PiperRT\n"
              << std::endl;

    print_cpu_info();

#if PIPER_HAVE_CUDA
    print_cuda_info(-1);
#else
    std::cout << "\n[GPU] CUDA not enabled (PIPER_HAVE_CUDA=0)\n";
#endif

    return 0;
}
