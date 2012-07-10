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

            void* data() { return 0; }
        };

        struct EmptyJob : public OneShotJob<void>
        {
            EmptyJob( AssignmentAlgorithm& algo, int masterRank ) :
                OneShotJob<void>( algo, masterRank, *(new DummyJobStore) )
                // FIXME memory leak => will be corrected by using const correctness
            {
                // empty
            }

            ~EmptyJob()
            {
                std::vector< int > idles = assignmentAlgo.idles();
                for(unsigned i = 0, size = idles.size(); i < size; ++i)
                {
                    comm.send( idles[i], Channel::Commands, Message::Kill );
                }
            }
        };
    }
}

# endif // __EO_TERMINATE_H__
