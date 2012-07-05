# ifndef __EO_TERMINATE_H__
# define __EO_TERMINATE_H__

# include "eoMpi.h"

namespace eo
{
    namespace mpi
    {
        struct DummySendTaskFunction : public SendTaskFunction<void>
        {
            void operator()( int _ )
            {
                ++_;
            }
        };

        struct DummyHandleResponseFunction : public HandleResponseFunction<void>
        {
            void operator()( int _ )
            {
                ++_;
            }
        };

        struct DummyProcessTaskFunction : public ProcessTaskFunction<void>
        {
            void operator()()
            {
                // nothing!
            }
        };

        struct DummyIsFinishedFunction : public IsFinishedFunction<void>
        {
            bool operator()()
            {
                return true;
            }
        };

        struct DummyJobStore : public JobStore<void>
        {
            using JobStore<void>::_stf;
            using JobStore<void>::_hrf;
            using JobStore<void>::_ptf;
            using JobStore<void>::_iff;

            DummyJobStore()
            {
                _stf = new DummySendTaskFunction;
                _hrf = new DummyHandleResponseFunction;
                _ptf = new DummyProcessTaskFunction;
                _iff = new DummyIsFinishedFunction;
            }

            ~DummyJobStore()
            {
                delete _stf;
                delete _hrf;
                delete _ptf;
                delete _iff;
            }

            void* data() { return 0; }
        };

        struct EmptyJob : public Job<void>
        {
            EmptyJob( AssignmentAlgorithm& algo, int masterRank ) :
                Job<void>( algo, masterRank, *(new DummyJobStore) )
                // FIXME memory leak => will be corrected by using const correctness
            {
                // empty
            }
        };

        /*
        class TerminateJob : public Job
        {
            public:
                TerminateJob( AssignmentAlgorithm& algo, int _ )
                    : Job( algo, _ )
                {
                    // empty
                }

                void sendTask( int wrkRank )
                {
                    // empty
                }

                void handleResponse( int wrkRank )
                {
                    // empty
                }

                void processTask( )
                {
                    // empty
                }

                bool isFinished()
                {
                    return true;
                }
        };
        */
    }
}

# endif // __EO_TERMINATE_H__
