#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Row
{
    int wrank;
    int ai, aj; // arithmetic mapping from world rank
    int grank;  // rank in grid_comm
    int gi, gj; // coords from MPI_Cart_coords
};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int wrank, wsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

    // Pr,Pc from args; e.g. mpirun -n 8 ./a.out 2 4
    if (argc < 3)
    {
        if (wrank == 0)
            std::fprintf(stderr, "Usage: %s Pr Pc\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    int Pr = std::atoi(argv[1]), Pc = std::atoi(argv[2]);
    if (Pr * Pc != wsize)
    {
        if (wrank == 0)
            std::fprintf(stderr, "Need Pr*Pc == world_size (%d)\n", wsize);
        MPI_Finalize();
        return 1;
    }

    // 1) Arithmetic "grid" (what you'd do without grid_comm)
    int ai = wrank / Pc, aj = wrank % Pc;

    // 2) Real MPI cartesian grid_comm
    int dims[2] = {Pr, Pc}, periods[2] = {0, 0};
    MPI_Comm grid = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder=*/1, &grid);

    int grank;
    MPI_Comm_rank(grid, &grank);
    int coords[2];
    MPI_Cart_coords(grid, grank, 2, coords);

    Row local{wrank, ai, aj, grank, coords[0], coords[1]};

    std::vector<Row> all;
    if (wrank == 0)
        all.resize(wsize);

    MPI_Gather(&local, sizeof(Row), MPI_BYTE,
               wrank == 0 ? all.data() : nullptr, sizeof(Row), MPI_BYTE,
               0, MPI_COMM_WORLD);

    if (wrank == 0)
    {
        std::printf("world_size=%d, Pr=%d Pc=%d, reorder=1\n", wsize, Pr, Pc);
        std::printf("wrank  arith(i,j)   grid_rank  cart(i,j)\n");
        std::printf("-----  ----------   ---------  ---------\n");
        for (const auto &r : all)
        {
            std::printf("%5d  (%2d,%2d)        %5d     (%2d,%2d)\n",
                        r.wrank, r.ai, r.aj, r.grank, r.gi, r.gj);
        }
        std::printf("\nIf MPI reorders ranks, arith(i,j) can differ from cart(i,j).\n");
        std::printf("Even if it doesn't, cart(i,j) proves MPI stores the coordinate mapping.\n");
    }

    MPI_Comm_free(&grid);
    MPI_Finalize();
    return 0;
}
