# ifndef __MPI_NODE_H__
# define __MPI_NODE_H__

# include <boost/mpi.hpp>
namespace bmpi = boost::mpi;

namespace eo
{
    namespace mpi
    {
        class Node
        {
            public:

                static void init( int argc, char** argv )
                {
                    static bmpi::environment env( argc, argv );
                }

                static bmpi::communicator& comm()
                {
                    return _comm;
                }

            protected:
                static bmpi::communicator _comm;
        };
    }
}
# endif // __MPI_NODE_H__

