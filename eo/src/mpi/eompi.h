# ifndef __EO_MPI_H__
# define __EO_MPI_H__

# include <vector>
# include <map>

# include <boost/mpi.hpp>
namespace mpi = boost::mpi;

# include "assignmentAlgorithm.h"

# include <iostream>
using namespace std;
// TODO TODOB comment!

namespace EoMpi
{
    namespace Channel
    {
        const int Commands = 0;
    }

    namespace Message
    {
        const int Continue = 0;
        const int Finish = 1;
    }
}

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

template< typename EOT >
class MpiJob
{
    public:

    MpiJob( std::vector< EOT > & _data, AssignmentAlgorithm& algo ) :
        data( _data ),
        comm( MpiNode::comm() ),
        assignmentAlgo( algo )
    {
        // empty
    }

    // master
    virtual void sendTask( int wrkRank, int index ) = 0;
    virtual void handleResponse( int wrkRank, int index ) = 0;
    // worker
    virtual void processTask( ) = 0;

    void master( )
    {
        for( int i = 0, size = data.size();
                i < size;
                ++i)
        {
            cout << "Beginning loop for i = " << i << endl;
            int assignee = assignmentAlgo.get( );
            cout << "Assignee : " << assignee << endl;
            while( assignee <= 0 )
            {
                cout << "Waitin' for node..." << endl;
                mpi::status status = comm.probe( mpi::any_source, mpi::any_tag );
                int wrkRank = status.source();
                cout << "Node " << wrkRank << " just terminated." << endl;
                handleResponse( wrkRank, assignedTasks[ wrkRank ] );
                assignmentAlgo.confirm( wrkRank );
                assignee = assignmentAlgo.get( );
            }
            cout << "Assignee found : " << assignee << endl;
            assignedTasks[ assignee ] = i;
            comm.send( assignee, EoMpi::Channel::Commands, EoMpi::Message::Continue );
            sendTask( assignee, i );
        }

        // frees all the idle workers
        int idle;
        vector<int> idles;
        while ( ( idle = assignmentAlgo.get( ) ) > 0 )
        {
            comm.send( idle, EoMpi::Channel::Commands, EoMpi::Message::Finish );
            idles.push_back( idle );
        }
        for (int i = 0; i < idles.size(); ++i)
        {
            assignmentAlgo.confirm( idles[i] );
        }

        // wait for all responses
        int wrkNb = comm.size() - 1;
        while( assignmentAlgo.size() != wrkNb )
        {
            mpi::status status = comm.probe( mpi::any_source, mpi::any_tag );
            int wrkRank = status.source();
            handleResponse( wrkRank, assignedTasks[ wrkRank ] );
            comm.send( wrkRank, EoMpi::Channel::Commands, EoMpi::Message::Finish );
            assignmentAlgo.confirm( wrkRank );
        }
    }

    void worker( )
    {
        int order;
        while( true )
        {
            comm.recv( 0, EoMpi::Channel::Commands, order );
            if ( order == EoMpi::Message::Finish )
            {
                return;
            } else
            {
                processTask( );
            }
        }
    }

protected:

    std::vector<EOT> & data;
    std::map< int /* worker rank */, int /* index in vector */> assignedTasks;
    AssignmentAlgorithm& assignmentAlgo;
    mpi::communicator& comm;
};

template< class EOT >
class Role
{
    public:
        Role( MpiJob<EOT> & job, bool master ) :
            _job( job ),
            _master( master )
        {
            // empty
        }

        bool master()
        {
            return _master;
        }

        virtual void run( )
        {
            if( _master )
            {
                _job.master( );
            } else
            {
                _job.worker( );
            }
        }

        virtual ~Role()
        {
            // empty
        }

    protected:
        bool _master;
        MpiJob<EOT> & _job;
};
# endif // __EO_MPI_H__
