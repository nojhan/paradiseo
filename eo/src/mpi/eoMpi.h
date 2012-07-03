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
            const int Messages = 1;
        }

        namespace Message
        {
            const int Continue = 0;
            const int Finish = 1;
        }


        template< typename JobData, typename Wrapped >
        struct SharedDataFunction
        {
            SharedDataFunction( Wrapped * w = 0 ) : _wrapped( w )
            {
                // empty
            }

            void wrapped( Wrapped * w )
            {
                _wrapped = w;
            }

            void data( JobData* _d )
            {
                d = _d;
                if( _wrapped )
                {
                    _wrapped->data( _d );
                }
            }

            protected:
            JobData* d;
            Wrapped* _wrapped;
        };

        template< typename JobData >
        struct SendTaskFunction : public eoUF<int, void>, public SharedDataFunction< JobData, SendTaskFunction<JobData> >
        {
            public:

            SendTaskFunction( SendTaskFunction<JobData>* w = 0 ) : SharedDataFunction<JobData, SendTaskFunction<JobData> >( w )
            {
                // empty
            }

            virtual ~SendTaskFunction() {}
        };

        template< typename JobData >
        struct HandleResponseFunction : public eoUF<int, void>, public SharedDataFunction< JobData, HandleResponseFunction<JobData> >
        {
            public:

            HandleResponseFunction( HandleResponseFunction<JobData>* w = 0 ) : SharedDataFunction<JobData, HandleResponseFunction<JobData> >( w )
            {
                // empty
            }

            virtual ~HandleResponseFunction() {}
        };

        template< typename JobData >
        struct ProcessTaskFunction : public eoF<void>, public SharedDataFunction< JobData, ProcessTaskFunction<JobData> >
        {
            public:

            ProcessTaskFunction( ProcessTaskFunction<JobData>* w = 0 ) : SharedDataFunction<JobData, ProcessTaskFunction<JobData> >( w )
            {
                // empty
            }

            virtual ~ProcessTaskFunction() {}
        };

        template< typename JobData >
        struct IsFinishedFunction : public eoF<bool>, public SharedDataFunction< JobData, IsFinishedFunction<JobData> >
        {
            public:

            IsFinishedFunction( IsFinishedFunction<JobData>* w = 0 ) : SharedDataFunction<JobData, IsFinishedFunction<JobData> >( w )
            {
                // empty
            }

            virtual ~IsFinishedFunction() {}
        };

        template< typename JobData >
        struct JobStore
        {
            JobStore(
                SendTaskFunction<JobData>* stf,
                HandleResponseFunction<JobData>* hrf,
                ProcessTaskFunction<JobData>* ptf,
                IsFinishedFunction<JobData>* iff
            ) :
                _stf( stf ), _hrf( hrf ), _ptf( ptf ), _iff( iff )
            {
                // empty
            }

            JobStore()
            {
                // empty
            }

            SendTaskFunction<JobData> & sendTask() { return *_stf; }
            HandleResponseFunction<JobData> & handleResponse() { return *_hrf; }
            ProcessTaskFunction<JobData> & processTask() { return *_ptf; }
            IsFinishedFunction<JobData> & isFinished() { return *_iff; }

            void sendTask( SendTaskFunction<JobData>* stf ) { _stf = stf; }
            void handleResponse( HandleResponseFunction<JobData>* hrf ) { _hrf = hrf; }
            void processTask( ProcessTaskFunction<JobData>* ptf ) { _ptf = ptf; }
            void isFinished( IsFinishedFunction<JobData>* iff ) { _iff = iff; }

            void wrapSendTask( SendTaskFunction<JobData>* stf )
            {
                if( stf )
                {
                    stf->wrapped( _stf );
                    _stf = stf;
                }
            }

            void wrapHandleResponse( HandleResponseFunction<JobData>* hrf )
            {
                if( hrf )
                {
                    hrf->wrapped( _hrf );
                    _hrf = hrf;
                }
            }

            void wrapProcessTask( ProcessTaskFunction<JobData>* ptf )
            {
                if( ptf )
                {
                    ptf->wrapped( _ptf );
                    _ptf = ptf;
                }
            }

            void wrapIsFinished( IsFinishedFunction<JobData>* iff )
            {
                if( iff )
                {
                    iff->wrapped( _iff );
                    _iff = iff;
                }
            }

            // TODO commenter : laissé à la couche d'en dessous car impossible d'initialiser une donnée membre d'une classe mère depuis une classe fille.
            virtual JobData* data() = 0;

            protected:

            // TODO commenter : Utiliser des pointeurs pour éviter d'écraser les fonctions wrappées
            SendTaskFunction< JobData >* _stf;
            HandleResponseFunction< JobData >* _hrf;
            ProcessTaskFunction< JobData >* _ptf;
            IsFinishedFunction< JobData >* _iff;
        };

        template< class JobData >
        class Job
        {
            public:
                Job( AssignmentAlgorithm& _algo, int _masterRank, JobStore<JobData> & store ) :
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

                    sendTask.data( store.data() );
                    handleResponse.data( store.data() );
                    processTask.data( store.data() );
                    isFinished.data( store.data() );
                }

            protected:

                SendTaskFunction<JobData> & sendTask;
                HandleResponseFunction<JobData> & handleResponse;
                ProcessTaskFunction<JobData> & processTask;
                IsFinishedFunction<JobData> & isFinished;

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

