# include "eoMpi.h"

// MpiNode* MpiNodeStore::singleton;
namespace eo
{
    namespace mpi
    {
        bmpi::communicator Node::_comm;
        eoTimerStat timerStat;
    }
}

