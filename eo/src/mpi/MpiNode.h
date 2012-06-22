# ifndef __MPI_NODE_H__
# define __MPI_NODE_H__

# include <boost/mpi.hpp>
namespace mpi = boost::mpi;

class MpiNode
{
    public:

    static void init( int argc, char** argv )
    {
        static mpi::environment env( argc, argv );
    }

    static mpi::communicator& comm()
    {
        return _comm;
    }

    protected:
    static mpi::communicator _comm;
};

# endif // __MPI_NODE_H__
