# ifndef __EO_PARALLEL_APPLY_H__
# define __EO_PARALLEL_APPLY_H__

# include "eompi.h"

# include <eoFunctor.h>
# include <vector>

template< typename EOT >
class ParallelApply : public MpiJob< EOT >
{
    public:

        ParallelApply( eoUF<EOT&, void> & _proc, std::vector<EOT>& _pop, AssignmentAlgorithm & algo ) :
            MpiJob<EOT>( _pop, algo ),
            func( _proc )
        {
            // empty
        }

        virtual void sendTask( int wrkRank, int index )
        {
            MpiJob<EOT>::comm.send( wrkRank, 1, MpiJob<EOT>::data[ index ] );
        }

        virtual void handleResponse( int wrkRank, int index )
        {
            MpiJob<EOT>::comm.recv( wrkRank, 1, MpiJob<EOT>::data[ index ] );
        }

        virtual void processTask( )
        {
            EOT ind;
            MpiJob<EOT>::comm.recv( 0, 1, ind );
            func( ind );
            MpiJob<EOT>::comm.send( 0, 1, ind );
        }

    protected:
        eoUF<EOT&, void>& func;
};

# endif // __EO_PARALLEL_APPLY_H__


