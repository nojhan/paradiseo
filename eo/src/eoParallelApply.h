# ifndef __EO_PARALLEL_APPLY_H__
# define __EO_PARALLEL_APPLY_H__

# include "eompi.h"

# include <eoFunctor.h>
# include <vector>

template< typename EOT >
class ParallelApply : public MpiJob< EOT >
{
    public:

        ParallelApply( eoUF<EOT&, void> & _proc, std::vector<EOT>& _pop ) :
            MpiJob<EOT>( _pop ),
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
            cout << "Receiving individual." << endl;
            MpiJob<EOT>::comm.recv( 0, 1, ind );
            cout << "Applying function." << endl;
            func( ind );
            cout << "Sending result." << endl;
            MpiJob<EOT>::comm.send( 0, 1, ind );
            cout << "Leaving processTask" << endl;
        }

    protected:
        eoUF<EOT&, void>& func;
};

# endif // __EO_PARALLEL_APPLY_H__


