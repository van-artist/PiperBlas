#pragma once

#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <iostream>
#include <string>
#include <thread>
#include <sstream>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <fstream>
#include <unistd.h>
#endif

void pi_free(void **p);
double wall_now();

static std::string trim(const std::string &s)
{
    size_t b = s.find_first_not_of(" \t\r\n");
    size_t e = s.find_last_not_of(" \t\r\n");
    if (b == std::string::npos)
        return "";
    return s.substr(b, e - b + 1);
}

static std::string read_first_match_linux_cpuinfo(const std::string &key)
{
#if defined(__linux__)
    std::ifstream fin("/proc/cpuinfo");
    std::string line;
    while (std::getline(fin, line))
    {
        auto pos = line.find(':');
        if (pos == std::string::npos)
            continue;
        auto k = trim(line.substr(0, pos));
        if (k == key)
            return trim(line.substr(pos + 1));
    }
#endif
    return "";
}

inline void print_cpu_info()
{
    std::cout << "\n[CPU]\n";

    unsigned int logical = std::thread::hardware_concurrency();
    if (logical == 0)
        logical = 1;

    std::string brand;
    int physical_cores = -1;
    double mhz = -1.0;

#if defined(_WIN32)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    physical_cores = -1;

#elif defined(__APPLE__)
    {
        char buf[256];
        size_t size = sizeof(buf);
        if (sysctlbyname("machdep.cpu.brand_string", &buf, &size, nullptr, 0) == 0)
            brand = std::string(buf);
    }
    {
        int v = 0;
        size_t size = sizeof(v);
        if (sysctlbyname("hw.physicalcpu", &v, &size, nullptr, 0) == 0)
            physical_cores = v;
    }
    {
        long long hz = 0;
        size_t size = sizeof(hz);
        if (sysctlbyname("hw.cpufrequency", &hz, &size, nullptr, 0) == 0)
            mhz = (double)hz / 1e6;
    }

#elif defined(__linux__)
    brand = read_first_match_linux_cpuinfo("model name");
    {
        std::string s = read_first_match_linux_cpuinfo("cpu MHz");
        if (!s.empty())
            mhz = std::stod(s);
    }
    {
        std::string s = read_first_match_linux_cpuinfo("cpu cores");
        if (!s.empty())
            physical_cores = std::stoi(s);
    }
#endif

    if (!brand.empty())
        std::cout << "  Model          : " << brand << "\n";
    std::cout << "  Logical Threads: " << logical << "\n";
    if (physical_cores > 0)
        std::cout << "  Physical Cores : " << physical_cores << "\n";
    if (mhz > 0)
        std::cout << "  Nominal Freq   : " << mhz << " MHz\n";

#if defined(__linux__)
    std::string cache = read_first_match_linux_cpuinfo("cache size");
    if (!cache.empty())
        std::cout << "  Cache (cpuinfo): " << cache << "\n";
#endif
}
