#include "utils.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void pi_free(void **p)
{
    if (p && *p)
    {
        std::free(*p);
        *p = nullptr;
    }
}

void print_cuda_important_attrs(int device)
{
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int maxThreadsPerBlock = 0;
    int maxBlockDimX = 0;
    int maxBlockDimY = 0;
    int maxBlockDimZ = 0;
    int maxGridDimX = 0;
    int maxGridDimY = 0;
    int maxGridDimZ = 0;
    int maxSharedMemPerBlock = 0;
    int maxSharedMemPerSM = 0;
    int maxRegsPerBlock = 0;
    int maxRegsPerSM = 0;
    int maxThreadsPerSM = 0;
    int maxBlocksPerSM = 0;
    int warpSize = 0;
    int smCount = 0;
    int l1GlobalSupported = 0;
    int l1LocalSupported = 0;

    CHECK_CUDA(cudaDeviceGetAttribute(&maxThreadsPerBlock,
                                      cudaDevAttrMaxThreadsPerBlock, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxBlockDimX,
                                      cudaDevAttrMaxBlockDimX, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxBlockDimY,
                                      cudaDevAttrMaxBlockDimY, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxBlockDimZ,
                                      cudaDevAttrMaxBlockDimZ, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxGridDimX,
                                      cudaDevAttrMaxGridDimX, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxGridDimY,
                                      cudaDevAttrMaxGridDimY, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxGridDimZ,
                                      cudaDevAttrMaxGridDimZ, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemPerBlock,
                                      cudaDevAttrMaxSharedMemoryPerBlock, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemPerSM,
                                      cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxRegsPerBlock,
                                      cudaDevAttrMaxRegistersPerBlock, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxRegsPerSM,
                                      cudaDevAttrMaxRegistersPerMultiprocessor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxThreadsPerSM,
                                      cudaDevAttrMaxThreadsPerMultiProcessor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxBlocksPerSM,
                                      cudaDevAttrMaxBlocksPerMultiprocessor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&warpSize,
                                      cudaDevAttrWarpSize, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&smCount,
                                      cudaDevAttrMultiProcessorCount, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&l1GlobalSupported,
                                      cudaDevAttrGlobalL1CacheSupported, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&l1LocalSupported,
                                      cudaDevAttrLocalL1CacheSupported, device));

    int ccMajor = prop.major;
    int ccMinor = prop.minor;
    size_t totalConstMem = prop.totalConstMem;
    int l2CacheSize = prop.l2CacheSize;
    int memClockKHz = prop.memoryClockRate;
    int memBusWidthBits = prop.memoryBusWidth;

    double memBandwidthGBs = 0.0;
    if (memClockKHz > 0 && memBusWidthBits > 0)
    {
        // 近似带宽：clock * 2(DDR) * busWidth / 8(byte)  转成 GB/s
        memBandwidthGBs =
            (double)memClockKHz * 1000.0 * 2.0 * (double)memBusWidthBits / 8.0 / 1e9;
    }

    // 打印头信息
    printf("CUDA Device %d: %s\n", device, prop.name);
    printf("Compute Capability: %d.%d\n\n", ccMajor, ccMinor);

    // 表格头
    printf("+-------------------------------+----------------------+-"
           "-------------------------------------------+\n");
    printf("| %-29s | %-20s | %-41s |\n",
           "Attribute", "Value", "Description");
    printf("+-------------------------------+----------------------+-"
           "-------------------------------------------+\n");

    auto row = [](const char *name, const char *value, const char *desc)
    {
        printf("| %-29s | %-20s | %-41s |\n", name, value, desc);
    };

    char buf[64];

    // 线程/Block/Grid 能力
    snprintf(buf, sizeof(buf), "%d", maxThreadsPerBlock);
    row("MaxThreadsPerBlock", buf, "最大线程数 / block");

    snprintf(buf, sizeof(buf), "%d,%d,%d",
             maxBlockDimX, maxBlockDimY, maxBlockDimZ);
    row("MaxBlockDim(X,Y,Z)", buf, "block 各维最大尺寸");

    snprintf(buf, sizeof(buf), "%d,%d,%d",
             maxGridDimX, maxGridDimY, maxGridDimZ);
    row("MaxGridDim(X,Y,Z)", buf, "grid 各维最大尺寸");

    // SM / warp 信息
    snprintf(buf, sizeof(buf), "%d", warpSize);
    row("WarpSize", buf, "一个 warp 的线程数");

    snprintf(buf, sizeof(buf), "%d", smCount);
    row("MultiProcessorCount", buf, "SM 数量");

    snprintf(buf, sizeof(buf), "%d", maxThreadsPerSM);
    row("MaxThreadsPerSM", buf, "每个 SM 最大驻留线程");

    snprintf(buf, sizeof(buf), "%d", maxBlocksPerSM);
    row("MaxBlocksPerSM", buf, "每个 SM 最大驻留 blocks");

    // Shared Memory / Registers
    snprintf(buf, sizeof(buf), "%d B", maxSharedMemPerBlock);
    row("MaxSharedMemPerBlock", buf, "每 block 可用 shared mem");

    snprintf(buf, sizeof(buf), "%d B", maxSharedMemPerSM);
    row("MaxSharedMemPerSM", buf, "每 SM 总 shared mem");

    snprintf(buf, sizeof(buf), "%d", maxRegsPerBlock);
    row("MaxRegsPerBlock", buf, "每 block 寄存器上限");

    snprintf(buf, sizeof(buf), "%d", maxRegsPerSM);
    row("MaxRegsPerSM", buf, "每 SM 寄存器总量");

    // 常量内存 / L2
    snprintf(buf, sizeof(buf), "%zu B", totalConstMem);
    row("TotalConstantMemory", buf, "__constant__ 总容量");

    snprintf(buf, sizeof(buf), "%d B", l2CacheSize);
    row("L2CacheSize", buf, "L2 cache 大小");

    // L1 支持情况
    snprintf(buf, sizeof(buf), "%s", l1GlobalSupported ? "Yes" : "No");
    row("GlobalL1CacheSupported", buf, "global 是否可走 L1");

    snprintf(buf, sizeof(buf), "%s", l1LocalSupported ? "Yes" : "No");
    row("LocalL1CacheSupported", buf, "local 是否可走 L1");

    // 显存带宽相关
    snprintf(buf, sizeof(buf), "%d kHz", memClockKHz);
    row("MemoryClockRate", buf, "显存时钟");

    snprintf(buf, sizeof(buf), "%d bits", memBusWidthBits);
    row("MemoryBusWidth", buf, "显存总线宽度");

    snprintf(buf, sizeof(buf), "%.2f GB/s", memBandwidthGBs);
    row("TheoreticalBandwidth", buf, "理论显存带宽(近似)");

    printf("+-------------------------------+----------------------+-"
           "-------------------------------------------+\n");
}
