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

            ~ProcessTaskParallelEval()
            {
            }
        };

        template< class EOT >
        struct ParallelEvalStore : public ParallelApplyStore< EOT >
        {
            using ParallelApplyStore<EOT>::wrapProcessTask;

            ParallelEvalStore(
                    eoUF<EOT&, void> & _proc,
                    int _masterRank,
                    int _packetSize = 1
                   ) :
                ParallelApplyStore< EOT >( _proc, *( new std::vector<EOT> ), _masterRank, _packetSize )
                // FIXME memory leak because of vector ==> use const correctness
            {
                wrapProcessTask( new ProcessTaskParallelEval<EOT> );
            }

            void data( std::vector<EOT>& _pop )
            {
                ParallelApplyStore<EOT>::_data.init( _pop );
            }
        };
    }
}
# endif // __EO_PARALLEL_APPLY_H__

