# ifndef __EO_PARALLEL_APPLY_H__
# define __EO_PARALLEL_APPLY_H__

# include "eompi.h"

# include <eoFunctor.h>
# include <vector>

template< typename EOT >
class ParallelApply : public MpiJob< EOT >
{
    public:
        using MpiJob<EOT>::comm;
        using MpiJob<EOT>::data;
        using MpiJob<EOT>::_masterRank;

        ParallelApply( eoUF<EOT&, void> & _proc, std::vector<EOT>& _pop, AssignmentAlgorithm & algo, int _masterRank ) :
            MpiJob<EOT>( _pop, algo, _masterRank ),
            func( _proc )
        {
            // empty
        }

        virtual void sendTask( int wrkRank, int index )
        {
            comm.send( wrkRank, 1, data[ index ] );
        }

        virtual void handleResponse( int wrkRank, int index )
        {
            comm.recv( wrkRank, 1, data[ index ] );
        }

        virtual void processTask( )
        {
            EOT ind;
            comm.recv( _masterRank, 1, ind );
            func( ind );
            comm.send( _masterRank, 1, ind );
        }

    protected:
        eoUF<EOT&, void>& func;
};

# endif // __EO_PARALLEL_APPLY_H__


