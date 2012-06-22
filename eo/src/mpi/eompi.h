# ifndef __EO_MPI_H__
# define __EO_MPI_H__

# include <vector>
# include <map>
# include <utils/eoLogger.h>

# include "MpiNode.h"
# include "assignmentAlgorithm.h"
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
class MpiJob
{
    public:

    MpiJob( AssignmentAlgorithm& _algo, int _masterRank ) :
        assignmentAlgo( _algo ),
        comm( MpiNode::comm() ),
        masterRank( _masterRank )
    {
        _isMaster = MpiNode::comm().rank() == _masterRank;
    }

    // master
    virtual bool isFinished() = 0;
    virtual void sendTask( int wrkRank ) = 0;
    virtual void handleResponse( int wrkRank ) = 0;
    // worker
    virtual void processTask( ) = 0;

    void master( )
    {
        int totalWorkers = assignmentAlgo.availableWorkers();
        eo::log << eo::debug;
        eo::log << "[M] Have " << totalWorkers << " workers." << std::endl;

        while( ! isFinished() )
        {
            int assignee = assignmentAlgo.get( );
            while( assignee <= 0 )
            {
                eo::log << "[M] Waitin' for node..." << std::endl;
                mpi::status status = comm.probe( mpi::any_source, mpi::any_tag );
                int wrkRank = status.source();
                eo::log << "[M] Node " << wrkRank << " just terminated." << std::endl;
                handleResponse( wrkRank );
                assignmentAlgo.confirm( wrkRank );
                assignee = assignmentAlgo.get( );
            }
            eo::log << "[M] Assignee : " << assignee << std::endl;
            comm.send( assignee, EoMpi::Channel::Commands, EoMpi::Message::Continue );
            sendTask( assignee );
        }

        eo::log << "[M] Frees all the idle." << std::endl;
        // frees all the idle workers
        std::vector<int> idles = assignmentAlgo.idles();
        for(unsigned int i = 0; i < idles.size(); ++i)
        {
            comm.send( idles[i], EoMpi::Channel::Commands, EoMpi::Message::Finish );
        }

        eo::log << "[M] Waits for all responses." << std::endl;
        // wait for all responses
        while( assignmentAlgo.availableWorkers() != totalWorkers )
        {
            mpi::status status = comm.probe( mpi::any_source, mpi::any_tag );
            int wrkRank = status.source();
            handleResponse( wrkRank );
            comm.send( wrkRank, EoMpi::Channel::Commands, EoMpi::Message::Finish );
            assignmentAlgo.confirm( wrkRank );
        }

        eo::log << "[M] Leaving master task." << std::endl;
    }

    void worker( )
    {
        int order;
        eo::log << eo::debug;
        while( true )
        {
            eo::log << "[W" << comm.rank() << "] Waiting for an order..." << std::endl;
            comm.recv( masterRank, EoMpi::Channel::Commands, order );
            if ( order == EoMpi::Message::Finish )
            {
                eo::log << "[W" << comm.rank() << "] Leaving worker task." << std::endl;
                return;
            } else
            {
                eo::log << "[W" << comm.rank() << "] Processing task..." << std::endl;
                processTask( );
            }
        }
    }

    void run( )
    {
        ( _isMaster ) ? master( ) : worker( );
    }

    bool isMaster( )
    {
        return _isMaster;
    }

protected:
    AssignmentAlgorithm& assignmentAlgo;
    mpi::communicator& comm;
    int masterRank;
    bool _isMaster;
};
# endif // __EO_MPI_H__

