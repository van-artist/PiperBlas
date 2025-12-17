#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

void *aligned_alloc64(std::size_t bytes);
void fill_random_double(double *x, std::size_t n, std::uint64_t *seed);

int parse_int_flag(int argc, char **argv, const char *key, int defv);

bool is_directory(const char *path);
bool is_regular_file(const char *path);
std::string join_path(const std::string &dir, const std::string &name);
std::vector<std::string> list_files_in_dir(const char *dir_path);
