#include "utils.h"

#include <cstdlib>

void pi_free(void **p)
{
    if (p && *p)
    {
        std::free(*p);
        *p = nullptr;
    }
}
