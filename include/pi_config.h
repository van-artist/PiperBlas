#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        int thread_num;

    } pi_config;

    void config_init();
    const pi_config *config();
    void config_destroy();

#ifdef __cplusplus
}
#endif