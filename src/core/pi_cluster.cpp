#include <mpi.h>
#include "pi_cluster.hpp"
#include "pi_type.hpp"

ClusterContext *ClusterContext::instance_ = nullptr;

void ClusterContext::Init(MPI_Comm world)
{
    if (instance_)
    {
        return;
    }

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        throw std::runtime_error("MPI must be initialized before ClusterContext::Init()");
    }

    instance_ = new ClusterContext(world);
}

ClusterContext &ClusterContext::instance()
{
    if (!instance_)
    {
        throw std::runtime_error("ClusterContext not initialized");
    }
    return *instance_;
}

ClusterContext::ClusterContext(MPI_Comm world)
    : world_(world)
{
    MPI_Comm_size(world_, &node_num_);
}

ClusterContext::~ClusterContext()
{
}
