#include <iostream>
#include "core/common.hpp"
#include "cuda/cuda_common.cuh"
#include "core/pi_cluster.hpp"

int main()
{
    std::cout << "Hello PiperRT\n"
              << std::endl;

    print_cpu_info();
    print_cuda_info(0);
    std::cout << ClusterContext::instance().node_num() << std::endl;
    return 0;
}
