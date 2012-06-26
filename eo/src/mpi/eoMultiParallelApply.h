
# ifndef __EO_MULTI_PARALLEL_APPLY_H__
# define __EO_MULTI_PARALLEL_APPLY_H__

# include "eoParallelApply.h"

namespace eo
{
    namespace mpi
    {
        template< typename EOT >
        class MultiParallelApply : public ParallelApply<EOT>
        {
            public:

                // using ParallelApply<EOT>::comm;
                using ParallelApply<EOT>::masterRank;

                MultiParallelApply(
                        eoUF<EOT&, void> & _proc,
                        std::vector<EOT>& _pop,
                        AssignmentAlgorithm & algo,
                        int _masterRank,
                        int _packetSize = 1,
                        long _maxTime = 0
                        ) :
                    ParallelApply<EOT>( _proc, _pop, algo, _masterRank, _packetSize, _maxTime )
                {
                    // empty
                }

                virtual void processTask( )
                {
                    int order = Message::Continue;
                    while( order != Message::Finish )
                    {
                        ParallelApply<EOT>::processTask( );
                        ParallelApply<EOT>::comm.recv( masterRank, Channel::Commands, order );
                    }
                }
        };
    }
}
# endif // __EO_PARALLEL_APPLY_H__

