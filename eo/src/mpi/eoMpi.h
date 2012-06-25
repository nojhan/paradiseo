# ifndef __EO_MPI_H__
# define __EO_MPI_H__

# include <vector>
# include <map>
# include <utils/eoLogger.h>

# include "eoMpiNode.h"
# include "eoMpiAssignmentAlgorithm.h"
// TODO TODOB comment!

namespace eo
{
    namespace mpi
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

        class Job
        {
            public:

                Job( AssignmentAlgorithm& _algo, int _masterRank ) :
                    assignmentAlgo( _algo ),
                    comm( Node::comm() ),
                    masterRank( _masterRank )
                {
                    _isMaster = Node::comm().rank() == _masterRank;
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
                    eo::log << "[M" << comm.rank() << "] Have " << totalWorkers << " workers." << std::endl;

                    while( ! isFinished() )
                    {
                        int assignee = assignmentAlgo.get( );
                        while( assignee <= 0 )
                        {
                            eo::log << "[M" << comm.rank() << "] Waitin' for node..." << std::endl;
                            bmpi::status status = comm.probe( bmpi::any_source, bmpi::any_tag );
                            int wrkRank = status.source();
                            eo::log << "[M" << comm.rank() << "] Node " << wrkRank << " just terminated." << std::endl;
                            handleResponse( wrkRank );
                            assignmentAlgo.confirm( wrkRank );
                            assignee = assignmentAlgo.get( );
                        }
                        eo::log << "[M" << comm.rank() << "] Assignee : " << assignee << std::endl;
                        comm.send( assignee, Channel::Commands, Message::Continue );
                        sendTask( assignee );
                    }

                    eo::log << "[M" << comm.rank() << "] Frees all the idle." << std::endl;
                    // frees all the idle workers
                    std::vector<int> idles = assignmentAlgo.idles();
                    for(unsigned int i = 0; i < idles.size(); ++i)
                    {
                        comm.send( idles[i], Channel::Commands, Message::Finish );
                    }

                    eo::log << "[M" << comm.rank() << "] Waits for all responses." << std::endl;
                    // wait for all responses
                    while( assignmentAlgo.availableWorkers() != totalWorkers )
                    {
                        bmpi::status status = comm.probe( bmpi::any_source, bmpi::any_tag );
                        int wrkRank = status.source();
                        handleResponse( wrkRank );
                        comm.send( wrkRank, Channel::Commands, Message::Finish );
                        assignmentAlgo.confirm( wrkRank );
                    }

                    eo::log << "[M" << comm.rank() << "] Leaving master task." << std::endl;
                }

                void worker( )
                {
                    int order;
                    eo::log << eo::debug;
                    while( true )
                    {
                        eo::log << "[W" << comm.rank() << "] Waiting for an order..." << std::endl;
                        comm.recv( masterRank, Channel::Commands, order );
                        if ( order == Message::Finish )
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
                bmpi::communicator& comm;
                int masterRank;
                bool _isMaster;
        };
    }
}

# endif // __EO_MPI_H__

