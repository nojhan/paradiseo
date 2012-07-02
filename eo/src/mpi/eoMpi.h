# ifndef __EO_MPI_H__
# define __EO_MPI_H__

# include <vector>
# include <map>
# include <sys/time.h>
# include <sys/resource.h>

# include <utils/eoLogger.h>
# include <utils/eoTimer.h>
# include <eoFunctor.h>
# include <eoExceptions.h>

# include "eoMpiNode.h"
# include "eoMpiAssignmentAlgorithm.h"

// TODO TODOB comment!

namespace eo
{
    namespace mpi
    {
        extern eoTimerStat timerStat;

        namespace Channel
        {
            const int Commands = 0;
        }

        namespace Message
        {
            const int Continue = 0;
            const int Finish = 1;
        }

        class SendTaskFunction : public eoUF<int, void>
        {
            public:
            virtual ~SendTaskFunction() {}
        };

        class HandleResponseFunction : public eoUF<int, void>
        {
            public:
            virtual ~HandleResponseFunction() {}
        };

        class ProcessTaskFunction : public eoF<void>
        {
            public:
            virtual ~ProcessTaskFunction() {}
        };

        class IsFinishedFunction : public eoF<bool>
        {
            public:
            virtual ~IsFinishedFunction() {}
        };

        struct JobStore
        {
            virtual SendTaskFunction & sendTask() const = 0;
            virtual HandleResponseFunction & handleResponse() const = 0;
            virtual ProcessTaskFunction & processTask() const = 0;
            virtual IsFinishedFunction & isFinished() const = 0;
        };

        class Job
        {
            public:
                Job( AssignmentAlgorithm& _algo, int _masterRank, const JobStore & store ) :
                // Job( AssignmentAlgorithm& _algo, int _masterRank, long maxTime = 0 ) :
                    assignmentAlgo( _algo ),
                    comm( Node::comm() ),
                    // _maxTime( maxTime ),
                    masterRank( _masterRank ),
                    // Functors
                    sendTask( store.sendTask() ),
                    handleResponse( store.handleResponse() ),
                    processTask( store.processTask() ),
                    isFinished( store.isFinished() )
                {
                    _isMaster = Node::comm().rank() == _masterRank;
                }

                /*
                // master
                virtual bool isFinished() = 0;
                virtual void sendTask( int wrkRank ) = 0;
                virtual void handleResponse( int wrkRank ) = 0;
                // worker
                virtual void processTask( ) = 0;
                */

            protected:

                SendTaskFunction & sendTask;
                HandleResponseFunction & handleResponse;
                ProcessTaskFunction & processTask;
                IsFinishedFunction & isFinished;

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
                        /*
                        getrusage( RUSAGE_SELF , &_usage );
                        _current = _usage.ru_utime.tv_sec + _usage.ru_stime.tv_sec;
                        if( _maxTime > 0 && _current > _maxTime )
                        {
                            timeStopped = true;
                            break;
                        }
                        */

                        timerStat.start("master_wait_for_assignee");
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
                        timerStat.stop("master_wait_for_assignee");
# ifndef NDEBUG
                        eo::log << "[M" << comm.rank() << "] Assignee : " << assignee << std::endl;
# endif

                        timerStat.start("master_wait_for_send");
                        comm.send( assignee, Channel::Commands, Message::Continue );
                        sendTask( assignee );
                        timerStat.stop("master_wait_for_send");
                    }

# ifndef NDEBUG
                    eo::log << "[M" << comm.rank() << "] Frees all the idle." << std::endl;
# endif
                    // frees all the idle workers
                    timerStat.start("master_wait_for_idles");
                    std::vector<int> idles = assignmentAlgo.idles();
                    for(unsigned int i = 0; i < idles.size(); ++i)
                    {
                        comm.send( idles[i], Channel::Commands, Message::Finish );
                    }
                    timerStat.stop("master_wait_for_idles");

# ifndef NDEBUG
                    eo::log << "[M" << comm.rank() << "] Waits for all responses." << std::endl;
# endif
                    // wait for all responses
                    timerStat.start("master_wait_for_all_responses");
                    while( assignmentAlgo.availableWorkers() != totalWorkers )
                    {
                        bmpi::status status = comm.probe( bmpi::any_source, bmpi::any_tag );
                        int wrkRank = status.source();
                        handleResponse( wrkRank );
                        comm.send( wrkRank, Channel::Commands, Message::Finish );
                        assignmentAlgo.confirm( wrkRank );
                    }
                    timerStat.stop("master_wait_for_all_responses");

# ifndef NDEBUG
                    eo::log << "[M" << comm.rank() << "] Leaving master task." << std::endl;
# endif
                    /*
                    if( timeStopped )
                    {
                        throw eoMaxTimeException( _current );
                    }
                    */
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
                        timerStat.start("worker_wait_for_order");
                        comm.recv( masterRank, Channel::Commands, order );
                        timerStat.stop("worker_wait_for_order");
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
                // const long _maxTime;
        };
    }
}

# endif // __EO_MPI_H__

