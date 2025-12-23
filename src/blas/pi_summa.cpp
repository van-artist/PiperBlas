#include <mpi.h>
#include "pi_type.hpp"

#include "core/common.hpp"
#include "core/pi_config.hpp"

#include "pi_blas.hpp"
#include "cuda/cuda_kernels.cuh"

// TODO: 单节点gemm后端
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
}

piState pi_summa_fp64(double *A, double *B, double *C,
                      double alpha, double beta,
                      int M, int K, int N)
{
    // TODO: 划分节点阵列

    // TODO: 组织通信组

    // TODO: 配置与索引确认

    // TODO:初始化分发数据

    return piSuccess;
}

piState pi_summa_fp32(float *A, float *B, float *C,
                      float alpha, float beta,
                      int m, int k, int n)
{
    // TODO: 划分节点阵列

    // TODO: 组织通信组

    // TODO: 配置与索引确认

    // TODO:初始化分发数据

    return piSuccess;
}
