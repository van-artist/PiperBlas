#include <mpi.h>
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
    int q;               // 总网格数
    int r;               // 网格行数
    int c;               // 网格列数
    int rank_world;      // 当前节点在world_comm中的rank
    int rank_grid;       // 当前节点在grid_comm中的rank
    int row;             // 当前节点所在行
    int col;             // 当前节点所在列
};

inline SummaGrid create_summa_grid(MPI_Comm world)
{
    SummaGrid sg;
    sg.world_comm = world;
    MPI_Comm_size(world, &sg.q);
    MPI_Comm_rank(world, &sg.rank_world);
    int dims[2] = {0, 0};
    MPI_Dims_create(sg.q, 2, dims);
    sg.r = dims[0];
    sg.c = dims[1];
    int periods[2] = {0};
    int reorder = 0;
    MPI_Cart_create(world, 2, dims, periods, reorder, &sg.grid_comm);
    MPI_Comm_rank(sg.grid_comm, &sg.rank_grid);
    int coords[2];
    MPI_Cart_coords(sg.grid_comm, sg.rank_grid, 2, coords);
    sg.row = coords[0];
    sg.col = coords[1];
    MPI_Comm_split(sg.grid_comm, sg.col, sg.row, &sg.col_comm);
    MPI_Comm_split(sg.grid_comm, sg.row, sg.col, &sg.row_comm);
    return sg;
}

inline piState nodeGemmFp32(float *A, float *B, float *C,
                            float alpha, float beta,
                            size_t M, size_t K, size_t N)
{
    return pi_cuda_gemm_fp32(A, B, C, alpha, beta, M, K, N);
}

inline piState nodeGemmFp64(double *A, double *B, double *C,
                            double alpha, double beta,
                            size_t M, size_t K, size_t N)
{
    return pi_cuda_gemm_fp64(A, B, C, alpha, beta, M, K, N);
}

// TODO: 发送单块子矩阵
template <typename T>
inline void send_block(MPI_Datatype dtype, T *start, int count)
{
}

// TODO: 接收单块子矩阵
template <typename T>
inline void recv_block(MPI_Datatype dtype, T *start, int count)
{
}

// MPI节点内进程
template <typename T>
piState pi_pdgmme(T *A, T *B, T *C, T alpha, T beta,    // 问题定义参数
                  int M, int K, int N,                  // 矩阵形状描述
                  int nb,                               // 总逻辑划分块数
                  int r,                                // 节点行数
                  int c,                                // 节点列数
                  MPI_Comm row_commm, MPI_Comm col_comm // 行与列通信域名
)
{

    // TODO:流水线填充阶段

    // TODO:流水线稳态阶段

    // TODO:流水线排出阶段
    return piSuccess;
}

template <typename T>
static inline piState pi_summa_impl(T *A, T *B, T *C,
                                    T alpha, T beta,
                                    int M, int K, int N, int p)
{
    // TODO: 初始化配置
    SummaGrid sg = create_summa_grid(MPI_COMM_WORLD);
    bool is_root = (sg.rank_world == 0);

    // TODO: 初始化分发数据

    // TODO: SUMMA 主循环

    return piSuccess;
}

piState pi_summa_fp64(double *A, double *B, double *C,
                      double alpha, double beta,
                      int M, int K, int N, int p)
{
    return pi_summa_impl<double>(A, B, C, alpha, beta, M, K, N, p);
}

piState pi_summa_fp32(float *A, float *B, float *C,
                      float alpha, float beta,
                      int m, int k, int n, int p)
{
    return pi_summa_impl<float>(A, B, C, alpha, beta, m, k, n, p);
}