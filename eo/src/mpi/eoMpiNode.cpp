# include "eoMpiNode.h"

namespace eo
{
    namespace mpi
    {
        void Node::init( int argc, char** argv )
        {
            static bmpi::environment env( argc, argv );
        }

        bmpi::communicator& Node::comm()
        {
            return _comm;
        }

        bmpi::communicator Node::_comm;
    }
}
