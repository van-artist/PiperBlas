#include "core/pi_config.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>

// 单例模式管理全局资源
static pi_config g_cfg;
static pthread_once_t g_cfg_once = PTHREAD_ONCE_INIT;

static int getenv_int(const char *name, int defval)
{
    const char *env = std::getenv(name);
    if (!env)
        return defval;

    char *endptr = nullptr;
    errno = 0;
    long val = std::strtol(env, &endptr, 10);
    if (errno || endptr == env || *endptr != '\0' || val <= 0)
    {
        std::fprintf(stderr, "[config] Invalid value for %s=\"%s\", using default %d\n",
                     name, env, defval);
        return defval;
    }
    return static_cast<int>(val);
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
