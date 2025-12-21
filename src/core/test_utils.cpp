#include "core/test_utils.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>

void *aligned_alloc64(std::size_t bytes)
{
    void *p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0 || !p)
    {
        std::fprintf(stderr, "alloc fail\n");
        std::exit(1);
    }
    return p;
}

void fill_random_double(double *x, std::size_t n, std::uint64_t *seed)
{
    for (std::size_t i = 0; i < n; i++)
    {
        *seed = (*seed * 2862933555777941757ULL) + 3037000493ULL;
        double v = ((double)(*seed >> 11) / (double)(1ULL << 53)) * 2.0 - 1.0;
        x[i] = v;
    }
}

int parse_int_flag(int argc, char **argv, const char *key, int defv)
{
    const std::size_t klen = std::strlen(key);
    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        if (std::strncmp(a, key, klen) == 0)
        {
            return std::atoi(a + klen);
        }
    }
    return defv;
}

bool is_directory(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        return false;
    return S_ISDIR(st.st_mode);
}

bool is_regular_file(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        return false;
    return S_ISREG(st.st_mode);
}

std::string join_path(const std::string &dir, const std::string &name)
{
    if (dir.empty())
        return name;
    if (dir.back() == '/')
        return dir + name;
    return dir + "/" + name;
}

std::vector<std::string> list_files_in_dir(const char *dir_path)
{
    std::vector<std::string> files;

    DIR *d = opendir(dir_path);
    if (!d)
        return files;

    while (true)
    {
        struct dirent *ent = readdir(d);
        if (!ent)
            break;

        const char *name = ent->d_name;
        if (!name || std::strcmp(name, ".") == 0 || std::strcmp(name, "..") == 0)
            continue;

        std::string full = join_path(dir_path, name);
        if (is_regular_file(full.c_str()))
            files.push_back(full);
    }

    closedir(d);
    std::sort(files.begin(), files.end());
    return files;
}
