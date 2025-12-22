#pragma once
#include <mpi.h>

class ClusterContext
{
public:
    static void Init(MPI_Comm world = MPI_COMM_WORLD);
    static ClusterContext &instance();

    int node_num() const { return node_num_; }

private:
    ClusterContext(MPI_Comm world);
    ~ClusterContext();

    MPI_Comm world_;
    int node_num_{1};

    static ClusterContext *instance_;
};
