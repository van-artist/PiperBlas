#include "core/common.hpp"

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


double wall_now()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}
