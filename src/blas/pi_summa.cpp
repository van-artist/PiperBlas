#include <mpi.h>
#include <limits>
#include <type_traits>
#include <vector>
#include <cstring>
#include "pi_type.hpp"

#include "core/common.hpp"
#include "core/pi_config.hpp"

#include "pi_blas.hpp"
#include "cuda/cuda_kernels.cuh"
#include "core/pi_cluster.hpp"

struct SummaGrid
{
    MPI_Comm world_comm; // 上层通信域，这里就是MPI_COMM_WORDL
    MPI_Comm grid_comm;  // 笛卡尔划分后的网格通信域名
    MPI_Comm row_comm;   // 节点所在的行通信域
    MPI_Comm col_comm;   // 节点所在的列通信域名
    int q;               // 总进程网格数
    int rows;            // 网格行数
    int cols;            // 网格列数
    int rank_world;      // 当前节点在world_comm中的rank
    int rank_grid;       // 当前节点在grid_comm中的rank
    int local_row;       // 当前节点所在行
    int local_col;       // 当前节点所在列
};

inline SummaGrid create_summa_grid(MPI_Comm world, int pr, int pc)
{
    SummaGrid sg;
    sg.world_comm = world;
    MPI_Comm_size(world, &sg.q);
    MPI_Comm_rank(world, &sg.rank_world);

    if (pr * pc != sg.q)
    {
        sg.grid_comm = MPI_COMM_NULL;
        sg.row_comm = MPI_COMM_NULL;
        sg.col_comm = MPI_COMM_NULL;
        sg.rows = sg.cols = 0;
        sg.rank_grid = -1;
        sg.local_row = sg.local_col = -1;
        return sg;
    }

    sg.rows = pr;
    sg.cols = pc;

    int dims[2] = {pr, pc};
    int periods[2] = {0, 0};
    int reorder = 0;

    MPI_Cart_create(world, 2, dims, periods, reorder, &sg.grid_comm);
    MPI_Comm_rank(sg.grid_comm, &sg.rank_grid);

    int coords[2];
    MPI_Cart_coords(sg.grid_comm, sg.rank_grid, 2, coords);
    sg.local_row = coords[0];
    sg.local_col = coords[1];

    MPI_Comm_split(sg.grid_comm, sg.local_col, sg.local_row, &sg.col_comm);
    MPI_Comm_split(sg.grid_comm, sg.local_row, sg.local_col, &sg.row_comm);

    return sg;
}

// 逻辑数据分块和线程分块分离的数据维度计算
static inline int numroc(int n, int nb,
                         int iproc, int isrcproc,
                         int nprocs)
{
    if (n <= 0)
        return 0;

    int nfull = n / nb;
    int last = n % nb;

    int mydist = (iproc - isrcproc + nprocs) % nprocs;

    int nlocal = (nfull / nprocs) * nb;

    int extras = nfull % nprocs;
    if (mydist < extras)
        nlocal += nb;
    else if (mydist == extras && last != 0)
        nlocal += last;

    return nlocal;
}

void ring_broadcast(MPI_Comm comm, void *buffer,
                    int count, MPI_Datatype datatype, int root)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    if (rank == root)
    {
        MPI_Send(buffer, count, datatype, next, 0, comm);
    }
    else
    {
        MPI_Recv(buffer, count, datatype, prev, 0, comm, MPI_STATUS_IGNORE);
        if (next != root)
        {
            MPI_Send(buffer, count, datatype, next, 0, comm);
        }
    }
}

template <typename T>
static void scatter_block_cyclic_2d(
    const T *global, int ld_global, // 全局矩阵的地址与步长
    T *local, int ld_local,         // 本地矩阵的地址与步长
    int M, int N,                   // 全局维度
    int mb, int nb,                 // 块大小
    const SummaGrid &sg,
    MPI_Datatype dtype)
{

    int rank = sg.rank_grid;
    int P = sg.q;

    // 计算每个 rank 的本地矩形维度
    std::vector<int> counts(P, 0), displs(P, 0);
    std::vector<int> mloc(P, 0), nloc(P, 0);

    for (int r = 0; r < P; ++r)
    {
        int rrow = r / sg.cols;
        int rcol = r % sg.cols;
        mloc[r] = numroc(M, mb, rrow, 0, sg.rows);
        nloc[r] = numroc(N, nb, rcol, 0, sg.cols);
        counts[r] = mloc[r] * nloc[r]; // 每个 rank 收到完整本地矩形（列主序连续）
    }

    for (int r = 1; r < P; ++r)
        displs[r] = displs[r - 1] + counts[r - 1];

    // root 打包：把每个 rank 的本地矩形数组写到 sendbuf 的对应切片
    std::vector<T> sendbuf;
    if (rank == 0)
    {
        size_t total = (size_t)displs[P - 1] + (size_t)counts[P - 1];
        sendbuf.assign(total, T(0));

        // 遍历全局块
        for (int bi = 0; bi * mb < M; ++bi)
        {
            for (int bj = 0; bj * nb < N; ++bj)
            {
                int owner_row = bi % sg.rows;
                int owner_col = bj % sg.cols;
                int owner_rank = owner_row * sg.cols + owner_col;

                int rows_blk = std::min(mb, M - bi * mb);
                int cols_blk = std::min(nb, N - bj * nb);

                // 该块在onwer_rank的本地块索引
                int lbi = bi / sg.rows;
                int lbj = bj / sg.cols;
                int li0 = lbi * mb;
                int lj0 = lbj * nb;

                int ld_owner = nloc[owner_rank];
                size_t base = (size_t)displs[owner_rank];

                for (int i = 0; i < rows_blk; ++i)
                {
                    for (int j = 0; j < cols_blk; ++j)
                    {
                        int gi = bi * mb + i;
                        int gj = bj * nb + j;

                        T val = global[(size_t)gi * (size_t)ld_global + (size_t)gj];

                        sendbuf[base + (size_t)(li0 + i) * (size_t)ld_owner + (size_t)(lj0 + j)] = val;
                    }
                }
            }
        }
    }

    int m_my = mloc[rank];
    int n_my = nloc[rank];
    int recvcount = m_my * n_my;

    std::vector<T> recvbuf((size_t)recvcount);

    MPI_Scatterv(
        (rank == 0 ? sendbuf.data() : nullptr),
        counts.data(), displs.data(), dtype,
        recvbuf.data(), recvcount, dtype,
        0, sg.grid_comm);

    if (ld_local == n_my)
    {
        memcpy(local, recvbuf.data(), (size_t)recvcount * sizeof(T));
    }
    else
    {
        for (int i = 0; i < m_my; ++i)
        {
            memcpy(
                local + (size_t)i * (size_t)ld_local,
                recvbuf.data() + (size_t)i * (size_t)n_my,
                (size_t)n_my * sizeof(T));
        }
    }
}
template <typename T>
void gather_block_cyclic_2d(const T *local, int ld_local,
                            T *global, int ld_global,
                            int M, int N,
                            int mb, int nb,
                            const SummaGrid &sg, MPI_Datatype dtype)
{
    if (mb <= 0 || nb <= 0)
        return;

    int m_loc = numroc(M, mb, sg.local_row, 0, sg.rows);
    int n_loc = numroc(N, nb, sg.local_col, 0, sg.cols);

    T *sendbuf = nullptr;
    size_t sendcount = (size_t)m_loc * (size_t)n_loc;

    if (sendcount > 0)
    {
        sendbuf = (T *)malloc(sendcount * sizeof(T));
        if (!sendbuf)
            return;

        if (ld_local == n_loc)
        {
            memcpy(sendbuf, local, sendcount * sizeof(T));
        }
        else
        {
            for (int i = 0; i < m_loc; ++i)
            {
                const T *src = local + (size_t)i * ld_local;
                T *dst = sendbuf + (size_t)i * n_loc;
                for (int j = 0; j < n_loc; ++j)
                    dst[j] = src[j];
            }
        }
    }

    int world_size = sg.rows * sg.cols;
    int *recvcounts = nullptr;
    int *displs = nullptr;
    T *recvbuf = nullptr;

    if (sg.rank_world == 0)
    {
        recvcounts = (int *)malloc((size_t)world_size * sizeof(int));
        displs = (int *)malloc((size_t)world_size * sizeof(int));
        if (!recvcounts || !displs)
        {
            free(recvcounts);
            free(displs);
            free(sendbuf);
            return;
        }

        long long total = 0;
        for (int r = 0; r < world_size; ++r)
        {
            int prow = r / sg.cols;
            int pcol = r % sg.cols;

            int mr = numroc(M, mb, prow, 0, sg.rows);
            int nr = numroc(N, nb, pcol, 0, sg.cols);

            long long cnt = (long long)mr * (long long)nr;
            if (cnt > 2147483647LL)
                cnt = 2147483647LL;
            recvcounts[r] = (int)cnt;
        }

        displs[0] = 0;
        for (int r = 1; r < world_size; ++r)
            displs[r] = displs[r - 1] + recvcounts[r - 1];

        total = (long long)displs[world_size - 1] + (long long)recvcounts[world_size - 1];
        if (total < 0)
            total = 0;

        if (total > 0)
        {
            recvbuf = (T *)malloc((size_t)total * sizeof(T));
            if (!recvbuf)
            {
                free(recvcounts);
                free(displs);
                free(sendbuf);
                return;
            }
        }
    }

    int scount_i = (int)sendcount;
    MPI_Gatherv(sendbuf, scount_i, dtype,
                recvbuf, recvcounts, displs, dtype,
                0, sg.grid_comm);

    if (sg.rank_world == 0 && recvbuf)
    {
        for (int r = 0; r < world_size; ++r)
        {
            int prow = r / sg.cols;
            int pcol = r % sg.cols;

            int mr = numroc(M, mb, prow, 0, sg.rows);
            int nr = numroc(N, nb, pcol, 0, sg.cols);

            const T *buf = recvbuf + displs[r];

            for (int il = 0; il < mr; ++il)
            {
                int block_i = il / mb;
                int off_i = il - block_i * mb;
                int ig = (block_i * sg.rows + prow) * mb + off_i;
                if (ig >= M)
                    continue;

                for (int jl = 0; jl < nr; ++jl)
                {
                    int block_j = jl / nb;
                    int off_j = jl - block_j * nb;
                    int jg = (block_j * sg.cols + pcol) * nb + off_j;
                    if (jg >= N)
                        continue;

                    global[(size_t)ig * ld_global + jg] = buf[(size_t)il * nr + jl];
                }
            }
        }
    }

    free(sendbuf);
    if (sg.rank_world == 0)
    {
        free(recvbuf);
        free(recvcounts);
        free(displs);
    }
}

template <typename T>
piState pi_pdgemm(T *A_loc, T *B_loc, T *C_loc,
                  T alpha, T beta,
                  int m_loc, int k_loc_A, int n_loc,
                  int K_global,
                  int nb,
                  int rows,
                  int cols,
                  MPI_Comm row_comm, MPI_Comm col_comm)
{
    MPI_Datatype dtype;
    if (std::is_same<T, float>::value)
        dtype = MPI_FLOAT;
    else if (std::is_same<T, double>::value)
        dtype = MPI_DOUBLE;
    else
        return piDataInvalid;

    int mycol = 0, myrow = 0;
    MPI_Comm_rank(row_comm, &mycol);
    MPI_Comm_rank(col_comm, &myrow);

    if (beta != T(1))
    {
        for (int i = 0; i < m_loc; ++i)
            for (int j = 0; j < n_loc; ++j)
                C_loc[(size_t)i * n_loc + j] *= beta;
    }

    const int kb_max = (nb > 0 ? nb : 1);
    T *A_panel = (T *)malloc((size_t)m_loc * (size_t)kb_max * sizeof(T));
    T *B_panel = (T *)malloc((size_t)kb_max * (size_t)n_loc * sizeof(T));
    if (!A_panel || !B_panel)
    {
        free(A_panel);
        free(B_panel);
        return piDataInvalid;
    }

    const int nsteps = (K_global + nb - 1) / nb;

    for (int b = 0; b < nsteps; ++b)
    {
        const int kb = std::min(nb, K_global - b * nb);

        const int owner_col = b % cols;
        const int owner_row = b % rows;

        T *Ap = A_panel;
        T *Bp = B_panel;

        if (mycol == owner_col)
        {
            const int kkA = (b / cols) * nb;
            for (int i = 0; i < m_loc; ++i)
            {
                const T *src = A_loc + (size_t)i * (size_t)k_loc_A + (size_t)kkA;
                T *dst = Ap + (size_t)i * (size_t)kb;
                for (int t = 0; t < kb; ++t)
                    dst[t] = src[t];
            }
        }

        if (myrow == owner_row)
        {
            const int kkB = (b / rows) * nb;
            for (int t = 0; t < kb; ++t)
            {
                const T *src = B_loc + (size_t)(kkB + t) * (size_t)n_loc;
                T *dst = Bp + (size_t)t * (size_t)n_loc;
                for (int j = 0; j < n_loc; ++j)
                    dst[j] = src[j];
            }
        }

        ring_broadcast(row_comm, Ap, m_loc * kb, dtype, owner_col);
        ring_broadcast(col_comm, Bp, kb * n_loc, dtype, owner_row);

        for (int i = 0; i < m_loc; ++i)
        {
            T *c = C_loc + (size_t)i * (size_t)n_loc;
            const T *a = Ap + (size_t)i * (size_t)kb;
            for (int t = 0; t < kb; ++t)
            {
                const T aval = alpha * a[t];
                const T *bb = Bp + (size_t)t * (size_t)n_loc;
                for (int j = 0; j < n_loc; ++j)
                    c[j] += aval * bb[j];
            }
        }
    }

    free(A_panel);
    free(B_panel);
    return piSuccess;
}

template <typename T>
static inline piState pi_summa_impl(T *A, T *B, T *C,
                                    T alpha, T beta,
                                    int M, int K, int N,
                                    int pr, int pc,
                                    int mb, int nb)
{
    // 初始化配置
    SummaGrid sg = create_summa_grid(MPI_COMM_WORLD, pr, pc);
    bool is_root = (sg.rank_world == 0);
    bool is_last_row = (sg.local_row == sg.rows - 1);
    bool is_last_col = (sg.local_col == sg.cols - 1);
    MPI_Datatype dtype;
    if (std::is_same<T, float>::value)
        dtype = MPI_FLOAT;
    else if (std::is_same<T, double>::value)
        dtype = MPI_DOUBLE;
    else
        return piDataInvalid;

    // 数据分发
    if (mb <= 0 || nb <= 0)
        return piDataInvalid;

    int m_loc = numroc(M, mb, sg.local_row, 0, sg.rows);   // A本地行数
    int k_loc_A = numroc(K, nb, sg.local_col, 0, sg.cols); // A本地列数
    int k_loc_B = numroc(K, mb, sg.local_row, 0, sg.rows); // B本地行数
    int n_loc = numroc(N, nb, sg.local_col, 0, sg.cols);   // B本地列数
    int m_loc_C = m_loc;                                   // C本地行数
    int n_loc_C = n_loc;                                   // C本地列数
    int ldA_loc = k_loc_A;                                 // A本地步长
    int ldB_loc = n_loc;                                   // B本地步长
    int ldC_loc = n_loc_C;                                 // C本地步长

    T *A_loc = (T *)malloc((size_t)m_loc * ldA_loc * sizeof(T));
    T *B_loc = (T *)malloc((size_t)k_loc_B * ldB_loc * sizeof(T));
    T *C_loc = (T *)malloc((size_t)m_loc_C * ldC_loc * sizeof(T));

    if (!A_loc || !B_loc || !C_loc)
    {
        free(A_loc);
        free(B_loc);
        free(C_loc);
        return piNoMemory;
    }

    scatter_block_cyclic_2d<T>(
        A, K,
        A_loc, ldA_loc,
        M, K,
        mb, nb,
        sg, dtype);

    scatter_block_cyclic_2d<T>(
        B, N,
        B_loc, ldB_loc,
        K, N,
        mb, nb,
        sg, dtype);

    if (beta != T(0))
    {
        scatter_block_cyclic_2d<T>(
            C, N,
            C_loc, ldC_loc,
            M, N,
            mb, nb,
            sg, dtype);
    }
    else
    {
        for (size_t i = 0; i < (size_t)m_loc_C * ldC_loc; ++i)
            C_loc[i] = T(0);
    }

    // 调用pdgemm
    pi_pdgemm(A_loc, B_loc, C_loc,
              alpha, beta,
              m_loc, k_loc_A, n_loc,
              K,
              nb,
              sg.rows, sg.cols,
              sg.row_comm, sg.col_comm);

    // 计算结果收集汇总
    gather_block_cyclic_2d<T>(
        C_loc, ldC_loc,
        C, N,
        M, N,
        mb, nb,
        sg, dtype);
    free(A_loc);
    free(B_loc);
    free(C_loc);

    return piSuccess;
}

piState pi_summa_fp64(double *A, double *B, double *C,
                      double alpha, double beta,
                      int M, int K, int N, int pr, int pc, int mb, int nb)
{
    return pi_summa_impl<double>(A, B, C, alpha, beta, M, K, N, pr, pc, mb, nb);
}

piState pi_summa_fp32(float *A, float *B, float *C,
                      float alpha, float beta,
                      int m, int k, int n, int pr, int pc, int mb, int nb)
{
    return pi_summa_impl<float>(A, B, C, alpha, beta, m, k, n, pr, pc, mb, nb);
}
