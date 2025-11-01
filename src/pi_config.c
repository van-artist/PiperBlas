#include "pi_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>

// 单例模式管理全局资源
static pi_config g_cfg;
static pthread_once_t g_cfg_once = PTHREAD_ONCE_INIT;

static int getenv_int(const char *name, int defval)
{
    const char *env = getenv(name);
    if (!env)
        return defval;

    char *endptr = NULL;
    errno = 0;
    long val = strtol(env, &endptr, 10);
    if (errno || endptr == env || *endptr != '\0' || val <= 0)
    {
        fprintf(stderr, "[config] Invalid value for %s=\"%s\", using default %d\n",
                name, env, defval);
        return defval;
    }
    return (int)val;
}

static void config_do_init()
{
    g_cfg.thread_num = getenv_int("PI_THREAD_NUM", 1);
}

void config_init()
{
    pthread_once(&g_cfg_once, config_do_init);
}

const pi_config *config()
{
    return &g_cfg;
}

void config_destroy()
{
}