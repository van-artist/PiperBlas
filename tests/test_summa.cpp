#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include "core/test_utils.hpp"
#include "pi_type.hpp"

piState pi_summa_fp64(double *A, double *B, double *C,
                      double alpha, double beta,
                      int M, int K, int N, int pr, int pc, int mb, int nb);

static inline double gflops_gemm(int M, int N, int K, double sec)
{
    if (sec <= 0.0)
        return 0.0;
    long double flops = 2.0L * (long double)M * (long double)N * (long double)K;
    return (double)(flops / sec / 1.0e9L);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int M = parse_int_flag(argc, argv, "--M=", 4096);
    int N = parse_int_flag(argc, argv, "--N=", 4096);
    int K = parse_int_flag(argc, argv, "--K=", 4096);
    int pr = parse_int_flag(argc, argv, "--pr=", 0);
    int pc = parse_int_flag(argc, argv, "--pc=", 0);
    int mb = parse_int_flag(argc, argv, "--mb=", 256);
    int nb = parse_int_flag(argc, argv, "--nb=", 256);
    int iters = parse_int_flag(argc, argv, "--iters=", 10);
    int warmup = parse_int_flag(argc, argv, "--warmup=", 1);

    if (pr <= 0 || pc <= 0)
    {
        int q = world_size;
        int p = (int)std::sqrt((double)q);
        while (p > 1 && q % p != 0)
            --p;
        pr = p;
        pc = q / p;
    }

    if (pr * pc != world_size)
    {
        if (world_rank == 0)
            std::fprintf(stderr, "Invalid grid: pr*pc=%d world_size=%d\n", pr * pc, world_size);
        MPI_Finalize();
        return 1;
    }

    double *A = nullptr;
    double *B = nullptr;
    double *C = nullptr;

    if (world_rank == 0)
    {
        A = (double *)aligned_alloc64((std::size_t)M * K * sizeof(double));
        B = (double *)aligned_alloc64((std::size_t)K * N * sizeof(double));
        C = (double *)aligned_alloc64((std::size_t)M * N * sizeof(double));

        std::uint64_t seed = 20260105ULL;
        fill_random_double(A, (std::size_t)M * K, &seed);
        fill_random_double(B, (std::size_t)K * N, &seed);
        fill_random_double(C, (std::size_t)M * N, &seed);
    }

    for (int w = 0; w < warmup; ++w)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        pi_summa_fp64(A, B, C, 1.0, 0.0, M, K, N, pr, pc, mb, nb);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    std::vector<double> max_times;
    max_times.reserve((std::size_t)iters);

    for (int it = 0; it < iters; ++it)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        piState st = pi_summa_fp64(A, B, C, 1.0, 0.0, M, K, N, pr, pc, mb, nb);

        double t1 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        if (st != piSuccess)
        {
            if (world_rank == 0)
                std::fprintf(stderr, "pi_summa_fp64 failed: %d\n", (int)st);
            MPI_Finalize();
            return 2;
        }

        double local = t1 - t0;
        double tmax = 0.0;
        MPI_Allreduce(&local, &tmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        max_times.push_back(tmax);

        if (world_rank == 0)
            std::printf("iter %d: time = %.6f s, GFLOPS = %.3f\n",
                        it, tmax, gflops_gemm(M, N, K, tmax));
    }

    double best = 1e100, worst = 0.0, sum = 0.0;
    for (double t : max_times)
    {
        best = std::min(best, t);
        worst = std::max(worst, t);
        sum += t;
    }
    double avg = sum / (double)iters;

    if (world_rank == 0)
    {
        TablePrinter tp(
            "SUMMA Benchmark",
            {"M", "N", "K", "pr", "pc", "mb", "nb", "iters", "warmup",
             "best_s", "avg_s", "worst_s",
             "best_GF", "avg_GF", "worst_GF"});

        tp.add_row({std::to_string(M),
                    std::to_string(N),
                    std::to_string(K),
                    std::to_string(pr),
                    std::to_string(pc),
                    std::to_string(mb),
                    std::to_string(nb),
                    std::to_string(iters),
                    std::to_string(warmup),
                    format_fixed(best, 10, 6),
                    format_fixed(avg, 10, 6),
                    format_fixed(worst, 10, 6),
                    format_fixed(gflops_gemm(M, N, K, best), 10, 3),
                    format_fixed(gflops_gemm(M, N, K, avg), 10, 3),
                    format_fixed(gflops_gemm(M, N, K, worst), 10, 3)});

        tp.print();

        std::free(A);
        std::free(B);
        std::free(C);
    }

    MPI_Finalize();
    return 0;
}
