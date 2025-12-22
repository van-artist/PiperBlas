#include <iostream>
#include "core/common.hpp"
#include "cuda/cuda_common.cuh"

int main()
{
    std::cout << "Hello PiperRT\n"
              << std::endl;

    print_cpu_info();
    print_cuda_info(0);
    return 0;
}
