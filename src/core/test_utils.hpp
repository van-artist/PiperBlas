#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/common.hpp"

void *aligned_alloc64(std::size_t bytes);
void fill_random_double(double *x, std::size_t n, std::uint64_t *seed);

int parse_int_flag(int argc, char **argv, const char *key, int defv);

bool is_directory(const char *path);
bool is_regular_file(const char *path);
std::string join_path(const std::string &dir, const std::string &name);
std::vector<std::string> list_files_in_dir(const char *dir_path);

// Measure average duration (ms) of repeated invocations of fn.
template <class F>
double time_avg_ms(int warmup, int iters, F fn)
{
    for (int i = 0; i < warmup; ++i)
        fn();
    double t0 = wall_now();
    for (int i = 0; i < iters; ++i)
        fn();
    double t1 = wall_now();
    return (t1 - t0) * 1e3 / (double)iters;
}

std::string basename_of(const std::string &path);
std::string format_fixed(double v, int width, int precision);
std::string format_int64(std::int64_t v, int width);
std::string format_scientific(double v, int width, int precision);

class TablePrinter
{
public:
    enum class Align
    {
        Left,
        Right
    };

    TablePrinter(std::string title,
                 std::vector<std::string> headers,
                 std::vector<Align> align = {});

    void add_row(std::vector<std::string> row);
    void print() const;

private:
    std::string title_;
    std::vector<std::string> headers_;
    std::vector<Align> align_;
    std::vector<std::vector<std::string>> rows_;
};
