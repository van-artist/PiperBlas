#include <mpi.h>
#include "pi_blas.hpp"
#include "core/pi_cluster.hpp"
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 1024;
    int M = N;
    int K = N;

    double *A_ptr = nullptr;
    double *B_ptr = nullptr;
    double *C_ptr = nullptr;

    std::vector<double> A, B, C; // 只有 rank0 才会真正分配

    if (rank == 0)
    {
        A.resize((size_t)M * K);
        B.resize((size_t)K * N);
        C.assign((size_t)M * N, 0.0);

        for (int i = 0; i < M * K; ++i)
            A[i] = (double)(i + 1);
        for (int i = 0; i < K * N; ++i)
            B[i] = (double)(i + 1);

        A_ptr = A.data();
        B_ptr = B.data();
        C_ptr = C.data();
    }

    double alpha = 1.0;
    double beta = 0.0;

    // 注意：所有 rank 都调用
    piState status = pi_summa_fp64(
        A_ptr, B_ptr, C_ptr,
        alpha, beta,
        M, K, N,
        size);

    MPI_Finalize();
    return 0;
}
