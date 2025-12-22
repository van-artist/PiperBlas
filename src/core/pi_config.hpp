#pragma once
#include <stdbool.h>

class PiConfig
{
public:
    static PiConfig &instance();

    int thread_num() const { return thread_num_; }

    PiConfig(const PiConfig &) = delete;
    PiConfig &operator=(const PiConfig &) = delete;

private:
    PiConfig();
    void load_from_env();
    int getenv_int(const char *name, int defval);

    int thread_num_{1};
};
