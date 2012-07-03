# ifndef __EO_MULTI_PARALLEL_APPLY_H__
# define __EO_MULTI_PARALLEL_APPLY_H__

# include "eoParallelApply.h"

namespace eo
{
    namespace mpi
    {
        template< class EOT >
        class ProcessTaskParallelEval : public ProcessTaskParallelApply<EOT>
        {
            public:

            using ProcessTaskParallelApply<EOT>::_wrapped;
            using ProcessTaskParallelApply<EOT>::d;

            void operator()()
            {
                int order = Message::Continue;
                while( order != Message::Finish )
                {
                    _wrapped->operator()();
                    d->comm.recv( d->masterRank, Channel::Commands, order );
                }
            }
        };

        template< class EOT >
        struct ParallelEvalStore : public ParallelApplyStore< EOT >
        {
            using ParallelApplyStore<EOT>::wrapProcessTask;

            ParallelEvalStore(
                    eoUF<EOT&, void> & _proc,
                    std::vector<EOT>& _pop,
                    int _masterRank,
                    // long _maxTime = 0,
                    int _packetSize = 1
                   ) :
                ParallelApplyStore< EOT >( _proc, _pop, _masterRank, _packetSize )
            {
                wrapProcessTask( new ProcessTaskParallelEval<EOT> );
            }
        };
    }
}
# endif // __EO_PARALLEL_APPLY_H__

