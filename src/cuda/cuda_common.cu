#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>
#include "core/common.h"

static std::string format_bytes(unsigned long long bytes)
{
       const char *suffix[] = {"B", "KiB", "MiB", "GiB", "TiB"};
       double v = (double)bytes;
       int i = 0;
       while (v >= 1024.0 && i < 4)
       {
              v /= 1024.0;
              i++;
       }

       char buf[64];
       if (i == 0)
              std::snprintf(buf, sizeof(buf), "%.0f %s", v, suffix[i]);
       else
              std::snprintf(buf, sizeof(buf), "%.2f %s", v, suffix[i]);
       return std::string(buf);
}

static std::string format_khz(int khz)
{
       if (khz <= 0)
              return "N/A";
       double mhz = khz / 1000.0;
       char buf[64];
       std::snprintf(buf, sizeof(buf), "%.1f MHz", mhz);
       return std::string(buf);
}

static std::string yesno(int v) { return v ? "Yes" : "No"; }

static void print_table_header(const char *title)
{
       std::printf("\n[%s]\n", title);
       std::printf("+-------------------------------+----------------------+-------------------------------------------+\n");
       std::printf("| %-29s | %-20s | %-41s |\n", "Attribute", "Value", "Description");
       std::printf("+-------------------------------+----------------------+-------------------------------------------+\n");
}

static void print_table_footer()
{
       std::printf("+-------------------------------+----------------------+-------------------------------------------+\n");
}

static void row(const char *name, const std::string &value, const char *desc)
{
       std::printf("| %-29s | %-20s | %-41s |\n", name, value.c_str(), desc);
}

static int get_attr_int(cudaDeviceAttr attr, int device)
{
       int v = 0;
       CHECK_CUDA(cudaDeviceGetAttribute(&v, attr, device));
       return v;
}

void print_cuda_info(int device)
{
       int count = 0;
       cudaError_t e = cudaGetDeviceCount(&count);

       if (e != cudaSuccess || count <= 0)
       {
              std::printf("\n[GPU]\n  CUDA device count: 0 (or CUDA init failed: %s)\n",
                          cudaGetErrorString(e));
              return;
       }

       int driver_ver = 0, runtime_ver = 0;
       CHECK_CUDA(cudaDriverGetVersion(&driver_ver));
       CHECK_CUDA(cudaRuntimeGetVersion(&runtime_ver));

       auto ver_to_str = [](int v)
       {
              int major = v / 1000;
              int minor = (v % 1000) / 10;
              char buf[32];
              std::snprintf(buf, sizeof(buf), "%d.%d", major, minor);
              return std::string(buf);
       };

       std::printf("\n[GPU]\n");
       std::printf("  CUDA Driver  : %s\n", ver_to_str(driver_ver).c_str());
       std::printf("  CUDA Runtime : %s\n", ver_to_str(runtime_ver).c_str());
       std::printf("  Device Count : %d\n", count);

       std::vector<int> devices;
       if (device < 0)
       {
              for (int i = 0; i < count; ++i)
                     devices.push_back(i);
       }
       else
       {
              if (device >= count)
              {
                     std::printf("  Requested device %d out of range [0, %d)\n", device, count);
                     return;
              }
              devices.push_back(device);
       }

       for (int dev : devices)
       {
              CHECK_CUDA(cudaSetDevice(dev));

              cudaDeviceProp prop{};
              CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

              int smCount = get_attr_int(cudaDevAttrMultiProcessorCount, dev);
              int warpSize = get_attr_int(cudaDevAttrWarpSize, dev);
              int maxThreadsPerBlock = get_attr_int(cudaDevAttrMaxThreadsPerBlock, dev);
              int maxThreadsPerSM = get_attr_int(cudaDevAttrMaxThreadsPerMultiProcessor, dev);
              int maxBlocksPerSM = get_attr_int(cudaDevAttrMaxBlocksPerMultiprocessor, dev);
              int maxSharedPerBlock = get_attr_int(cudaDevAttrMaxSharedMemoryPerBlock, dev);
              int maxSharedPerSM = get_attr_int(cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
              int regsPerBlock = get_attr_int(cudaDevAttrMaxRegistersPerBlock, dev);
              int regsPerSM = get_attr_int(cudaDevAttrMaxRegistersPerMultiprocessor, dev);
              int l2CacheSize = prop.l2CacheSize;

              int maxBlockDimX = get_attr_int(cudaDevAttrMaxBlockDimX, dev);
              int maxBlockDimY = get_attr_int(cudaDevAttrMaxBlockDimY, dev);
              int maxBlockDimZ = get_attr_int(cudaDevAttrMaxBlockDimZ, dev);
              int maxGridDimX = get_attr_int(cudaDevAttrMaxGridDimX, dev);
              int maxGridDimY = get_attr_int(cudaDevAttrMaxGridDimY, dev);
              int maxGridDimZ = get_attr_int(cudaDevAttrMaxGridDimZ, dev);

              int l1GlobalSupported = get_attr_int(cudaDevAttrGlobalL1CacheSupported, dev);
              int l1LocalSupported = get_attr_int(cudaDevAttrLocalL1CacheSupported, dev);
              int unifiedAddressing = get_attr_int(cudaDevAttrUnifiedAddressing, dev);
              int managedMemory = get_attr_int(cudaDevAttrManagedMemory, dev);

              int eccEnabled = 0;
#if CUDART_VERSION >= 4020
              eccEnabled = prop.ECCEnabled;
#endif

              int memClockKHz = prop.memoryClockRate;
              int memBusWidth = prop.memoryBusWidth;

              double memBandwidthGBs = 0.0;
              if (memClockKHz > 0 && memBusWidth > 0)
              {
                     memBandwidthGBs =
                         (double)memClockKHz * 1000.0 * 2.0 * (double)memBusWidth / 8.0 / 1e9;
              }

              int pciBus = -1, pciDev = -1, pciDomain = -1;
              cudaDeviceGetAttribute(&pciBus, cudaDevAttrPciBusId, dev);
              cudaDeviceGetAttribute(&pciDev, cudaDevAttrPciDeviceId, dev);
#if defined(cudaDevAttrPciDomainId)
              cudaDeviceGetAttribute(&pciDomain, cudaDevAttrPciDomainId, dev);
#endif

              std::printf("\nCUDA Device %d: %s\n", dev, prop.name);
              std::printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);

              if (pciBus >= 0 && pciDev >= 0)
              {
                     if (pciDomain >= 0)
                            std::printf("  PCIe: %04x:%02x:%02x\n", pciDomain, pciBus, pciDev);
                     else
                            std::printf("  PCIe: %02x:%02x\n", pciBus, pciDev);
              }

              print_table_header("CUDA Device Properties");

              row("TotalGlobalMemory", format_bytes((unsigned long long)prop.totalGlobalMem), "总显存容量");
              row("TotalConstantMemory", format_bytes((unsigned long long)prop.totalConstMem), "__constant__ 总容量");
              row("SharedMemPerBlock", format_bytes((unsigned long long)maxSharedPerBlock), "每 block 可用 shared mem");
              row("SharedMemPerSM", format_bytes((unsigned long long)maxSharedPerSM), "每 SM 总 shared mem");

              row("SMCount", std::to_string(smCount), "SM 数量");
              row("WarpSize", std::to_string(warpSize), "一个 warp 的线程数");
              row("MaxThreadsPerBlock", std::to_string(maxThreadsPerBlock), "最大线程数 / block");
              row("MaxThreadsPerSM", std::to_string(maxThreadsPerSM), "每个 SM 最大驻留线程");
              row("MaxBlocksPerSM", std::to_string(maxBlocksPerSM), "每个 SM 最大驻留 blocks");

              {
                     char buf[64];
                     std::snprintf(buf, sizeof(buf), "%d,%d,%d", maxBlockDimX, maxBlockDimY, maxBlockDimZ);
                     row("MaxBlockDim(X,Y,Z)", buf, "block 各维最大尺寸");
              }
              {
                     char buf[64];
                     std::snprintf(buf, sizeof(buf), "%d,%d,%d", maxGridDimX, maxGridDimY, maxGridDimZ);
                     row("MaxGridDim(X,Y,Z)", buf, "grid 各维最大尺寸");
              }

              row("RegsPerBlock", std::to_string(regsPerBlock), "每 block 寄存器上限");
              row("RegsPerSM", std::to_string(regsPerSM), "每 SM 寄存器总量");

              row("L2CacheSize", format_bytes((unsigned long long)l2CacheSize), "L2 cache 大小");

              row("GlobalL1CacheSupported", yesno(l1GlobalSupported), "global 是否可走 L1");
              row("LocalL1CacheSupported", yesno(l1LocalSupported), "local 是否可走 L1");

              row("UnifiedAddressing", yesno(unifiedAddressing), "统一虚拟寻址 UVA");
              row("ManagedMemory", yesno(managedMemory), "是否支持 managed memory");
              row("ECCEnabled", yesno(eccEnabled), "是否启用 ECC");

              row("GPUClockRate", format_khz(prop.clockRate), "核心时钟");
              row("MemClockRate", format_khz(memClockKHz), "显存时钟");
              {
                     char buf[64];
                     std::snprintf(buf, sizeof(buf), "%d bits", memBusWidth);
                     row("MemBusWidth", buf, "显存总线宽度");
              }
              {
                     char buf[64];
                     std::snprintf(buf, sizeof(buf), "%.2f GB/s", memBandwidthGBs);
                     row("TheoreticalBandwidth", buf, "理论显存带宽");
              }

              row("AsyncEngineCount", std::to_string(prop.asyncEngineCount), "拷贝引擎数量");
              row("ConcurrentKernels", yesno(prop.concurrentKernels), "是否支持并发 kernel");
              row("CanMapHostMemory", yesno(prop.canMapHostMemory), "是否支持零拷贝映射");
              row("ComputeMode", std::to_string(prop.computeMode), "Compute Mode 枚举值");

              print_table_footer();
       }
}
