#include "core/pi_config.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>

int PiConfig::getenv_int(const char *name, int defval)
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

PiConfig::PiConfig() { load_from_env(); }

void PiConfig::load_from_env() { thread_num_ = getenv_int("PI_THREAD_NUM", 1); }

PiConfig &PiConfig::instance()
{
    static PiConfig cfg;
    return cfg;
}
