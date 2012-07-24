# include "eoMpi.h"

namespace eo
{
    namespace mpi
    {
        bmpi::communicator Node::_comm;
        eoTimerStat timerStat;
    }
}

namespace mpi
{
    void broadcast( communicator & comm, int value, int root )
    {
        comm; // unused
        MPI_Bcast( &value, 1, MPI_INT, root, MPI_COMM_WORLD );
    }
}
