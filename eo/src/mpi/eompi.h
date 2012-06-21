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

struct BaseContinuator
{
    virtual bool operator()() = 0;
};

// template< typename EOT >
class MpiJob
{
    public:

    MpiJob( AssignmentAlgorithm& algo, BaseContinuator* c, int masterRank ) :
        assignmentAlgo( algo ),
        comm( MpiNode::comm() ),
        _masterRank( masterRank ),
        _continuator( c )
    {
        // empty
    }

    ~MpiJob()
    {
        delete _continuator;
    }

    // master
    virtual void sendTask( int wrkRank ) = 0;
    virtual void handleResponse( int wrkRank ) = 0;
    // worker
    virtual void processTask( ) = 0;

    void master( )
    {
        int totalWorkers = assignmentAlgo.size();
        cout << "[M] Have " << totalWorkers << " workers." << endl;

        while( (*_continuator)() )
        {
            int assignee = assignmentAlgo.get( );
            cout << "[M] Assignee : " << assignee << endl;
            while( assignee <= 0 )
            {
                cout << "[M] Waitin' for node..." << endl;
                mpi::status status = comm.probe( mpi::any_source, mpi::any_tag );
                int wrkRank = status.source();
                cout << "[M] Node " << wrkRank << " just terminated." << endl;
                handleResponse( wrkRank );
                assignmentAlgo.confirm( wrkRank );
                assignee = assignmentAlgo.get( );
            }
            comm.send( assignee, EoMpi::Channel::Commands, EoMpi::Message::Continue );
            sendTask( assignee );
        }

        /*
        for( int i = 0, size = data.size();
                i < size;
                ++i)
        {
            cout << "[M] Beginning loop for i = " << i << endl;
            int assignee = assignmentAlgo.get( );
            cout << "[M] Assignee : " << assignee << endl;
            while( assignee <= 0 )
            {
                cout << "[M] Waitin' for node..." << endl;
                mpi::status status = comm.probe( mpi::any_source, mpi::any_tag );
                int wrkRank = status.source();
                cout << "[M] Node " << wrkRank << " just terminated." << endl;
                handleResponse( wrkRank, assignedTasks[ wrkRank ] );
                assignmentAlgo.confirm( wrkRank );
                assignee = assignmentAlgo.get( );
            }
            cout << "[M] Assignee found : " << assignee << endl;
            assignedTasks[ assignee ] = i;
            comm.send( assignee, EoMpi::Channel::Commands, EoMpi::Message::Continue );
            sendTask( assignee, i );
        }
        */

        cout << "[M] Frees all the idle." << endl;
        // frees all the idle workers
        int idle;
        vector<int> idles;
        while ( ( idle = assignmentAlgo.get( ) ) > 0 )
        {
            comm.send( idle, EoMpi::Channel::Commands, EoMpi::Message::Finish );
            idles.push_back( idle );
        }
        for (unsigned int i = 0; i < idles.size(); ++i)
        {
            assignmentAlgo.confirm( idles[i] );
        }

        cout << "[M] Waits for all responses." << endl;
        // wait for all responses
        while( assignmentAlgo.size() != totalWorkers )
        {
            mpi::status status = comm.probe( mpi::any_source, mpi::any_tag );
            int wrkRank = status.source();
            handleResponse( wrkRank );
            comm.send( wrkRank, EoMpi::Channel::Commands, EoMpi::Message::Finish );
            assignmentAlgo.confirm( wrkRank );
        }

        cout << "[M] Leaving master task." << endl;
    }

    void worker( )
    {
        int order;
        while( true )
        {
            cout << "[W] Waiting for an order..." << std::endl;
            comm.recv( _masterRank, EoMpi::Channel::Commands, order );
            if ( order == EoMpi::Message::Finish )
            {
                return;
            } else
            {
                cout << "[W] Processing task..." << std::endl;
                processTask( );
            }
        }
    }

    int masterRank()
    {
        return _masterRank;
    }

protected:
    AssignmentAlgorithm& assignmentAlgo;
    BaseContinuator* _continuator;
    mpi::communicator& comm;
    int _masterRank;
};

class Role
{
    public:
        Role( MpiJob & job ) :
            _job( job )
        {
            _master = job.masterRank() == MpiNode::comm().rank();
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
        MpiJob & _job;
        bool _master;
};
# endif // __EO_MPI_H__
