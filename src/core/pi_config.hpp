#pragma once
#include <stdbool.h>

typedef struct
{
    int thread_num;

} pi_config;

void config_init();
const pi_config *config();
void config_destroy();
