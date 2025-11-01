#include "utils.h"
#include "stdlib.h"

void pi_free(void **p)
{
    if (*p)
    {
        free(*p);
        *p = NULL;
    }
}