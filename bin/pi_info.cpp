#include <iostream>
#include "core/common.h"

int main()
{

    std::cout << "Hello PiperRT" << std::endl;
#if PIPER_HAVE_CUDA
    print_cuda_info();
#endif

    return 0;
}