#include "core/test_utils.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <utility>
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
    fill_random<double>(x, n, seed);
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

std::vector<std::string> collect_inputs_or_exit(const char *path, const char *program_name)
{
    std::vector<std::string> inputs;
    if (is_directory(path))
    {
        inputs = list_files_in_dir(path);
    }
    else if (is_regular_file(path))
    {
        inputs.push_back(path);
    }

    if (inputs.empty())
    {
        std::fprintf(stderr, "%s: 输入路径无效或为空: %s\n", program_name ? program_name : "program", path);
        std::exit(1);
    }
    return inputs;
}

std::string basename_of(const std::string &path)
{
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos)
        return path;
    return path.substr(pos + 1);
}

std::string format_fixed(double v, int width, int precision)
{
    std::ostringstream oss;
    oss << std::fixed << std::setw(width) << std::setprecision(precision) << v;
    return oss.str();
}

std::string format_int64(std::int64_t v, int width)
{
    std::ostringstream oss;
    oss << std::setw(width) << v;
    return oss.str();
}

std::string format_scientific(double v, int width, int precision)
{
    std::ostringstream oss;
    oss << std::scientific << std::setw(width) << std::setprecision(precision) << v;
    return oss.str();
}

TablePrinter::TablePrinter(std::string title,
                           std::vector<std::string> headers,
                           std::vector<Align> align)
    : title_(std::move(title)), headers_(std::move(headers)), align_(std::move(align))
{
    if (align_.empty())
    {
        align_.resize(headers_.size(), Align::Right);
        if (!align_.empty())
            align_[0] = Align::Left;
    }
}

void TablePrinter::add_row(std::vector<std::string> row)
{
    rows_.push_back(std::move(row));
}

void TablePrinter::print() const
{
    if (headers_.empty())
        return;

    std::vector<std::size_t> widths(headers_.size(), 0);
    for (std::size_t i = 0; i < headers_.size(); ++i)
        widths[i] = std::max(widths[i], headers_[i].size());
    for (const auto &row : rows_)
        for (std::size_t i = 0; i < widths.size() && i < row.size(); ++i)
            widths[i] = std::max(widths[i], row[i].size());

    auto print_row = [&](const std::vector<std::string> &row)
    {
        for (std::size_t i = 0; i < widths.size(); ++i)
        {
            const std::string cell = (i < row.size()) ? row[i] : "";
            bool left = (i < align_.size() && align_[i] == Align::Left);
            std::printf(left ? "%-*s" : "%*s", (int)widths[i], cell.c_str());
            if (i + 1 < widths.size())
                std::printf(" | ");
        }
        std::printf("\n");
    };

    if (!title_.empty())
        std::printf("%s\n", title_.c_str());

    print_row(headers_);
    for (const auto &row : rows_)
        print_row(row);
}
