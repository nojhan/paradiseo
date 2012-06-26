# ifndef __EO_MPI_H__
# define __EO_MPI_H__

# include <vector>
# include <map>
# include <sys/time.h>
# include <sys/resource.h>

# include <utils/eoLogger.h>
# include <eoExceptions.h>

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

                Job( AssignmentAlgorithm& _algo, int _masterRank, long maxTime = 0 ) :
                    assignmentAlgo( _algo ),
                    comm( Node::comm() ),
                    masterRank( _masterRank ),
                    _maxTime( maxTime )
                {
                    _isMaster = Node::comm().rank() == _masterRank;
                }

                // master
                virtual bool isFinished() = 0;
                virtual void sendTask( int wrkRank ) = 0;
                virtual void handleResponse( int wrkRank ) = 0;
                // worker
                virtual void processTask( ) = 0;

            protected:

                void master( )
                {
                    int totalWorkers = assignmentAlgo.availableWorkers();
# ifndef NDEBUG
                    eo::log << eo::debug;
                    eo::log << "[M" << comm.rank() << "] Have " << totalWorkers << " workers." << std::endl;
# endif
                    bool timeStopped = false;
                    while( ! isFinished() )
                    {
                        // Time restrictions
                        getrusage( RUSAGE_SELF , &_usage );
                        _current = _usage.ru_utime.tv_sec + _usage.ru_stime.tv_sec;
                        if( _maxTime > 0 && _current > _maxTime )
                        {
                            timeStopped = true;
                            break;
                        }

                        int assignee = assignmentAlgo.get( );
                        while( assignee <= 0 )
                        {
# ifndef NDEBUG
                            eo::log << "[M" << comm.rank() << "] Waitin' for node..." << std::endl;
# endif
                            bmpi::status status = comm.probe( bmpi::any_source, bmpi::any_tag );
                            int wrkRank = status.source();
# ifndef NDEBUG
                            eo::log << "[M" << comm.rank() << "] Node " << wrkRank << " just terminated." << std::endl;
# endif
                            handleResponse( wrkRank );
                            assignmentAlgo.confirm( wrkRank );
                            assignee = assignmentAlgo.get( );
                        }
# ifndef NDEBUG
                        eo::log << "[M" << comm.rank() << "] Assignee : " << assignee << std::endl;
# endif

                        comm.send( assignee, Channel::Commands, Message::Continue );
                        sendTask( assignee );
                    }

# ifndef NDEBUG
                    eo::log << "[M" << comm.rank() << "] Frees all the idle." << std::endl;
# endif
                    // frees all the idle workers
                    std::vector<int> idles = assignmentAlgo.idles();
                    for(unsigned int i = 0; i < idles.size(); ++i)
                    {
                        comm.send( idles[i], Channel::Commands, Message::Finish );
                    }

# ifndef NDEBUG
                    eo::log << "[M" << comm.rank() << "] Waits for all responses." << std::endl;
# endif
                    // wait for all responses
                    while( assignmentAlgo.availableWorkers() != totalWorkers )
                    {
                        bmpi::status status = comm.probe( bmpi::any_source, bmpi::any_tag );
                        int wrkRank = status.source();
                        handleResponse( wrkRank );
                        comm.send( wrkRank, Channel::Commands, Message::Finish );
                        assignmentAlgo.confirm( wrkRank );
                    }

# ifndef NDEBUG
                    eo::log << "[M" << comm.rank() << "] Leaving master task." << std::endl;
# endif
                    if( timeStopped )
                    {
                        throw eoMaxTimeException( _current );
                    }
                }

                void worker( )
                {
                    int order;
# ifndef NDEBUG
                    eo::log << eo::debug;
# endif
                    while( true )
                    {
# ifndef NDEBUG
                        eo::log << "[W" << comm.rank() << "] Waiting for an order..." << std::endl;
# endif
                        comm.recv( masterRank, Channel::Commands, order );
                        if ( order == Message::Finish )
                        {
# ifndef NDEBUG
                            eo::log << "[W" << comm.rank() << "] Leaving worker task." << std::endl;
# endif
                            return;
                        } else
                        {
# ifndef NDEBUG
                            eo::log << "[W" << comm.rank() << "] Processing task..." << std::endl;
# endif
                            processTask( );
                        }
                    }
                }

            public:

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

                struct rusage _usage;
                long _current;
                const long _maxTime;
        };
    }
}

# endif // __EO_MPI_H__

