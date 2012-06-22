# ifndef __EO_PARALLEL_APPLY_H__
# define __EO_PARALLEL_APPLY_H__

# include "eompi.h"

# include <eoFunctor.h>
# include <vector>

template< typename EOT >
class ParallelApply : public MpiJob
{
    public:

        ParallelApply( eoUF<EOT&, void> & _proc, std::vector<EOT>& _pop, AssignmentAlgorithm & algo, int _masterRank ) :
            MpiJob( algo, _masterRank ),
            func( _proc ),
            index( 0 ),
            size( _pop.size() ),
            data( _pop )
        {
            // empty
        }

        virtual void sendTask( int wrkRank )
        {
            assignedTasks[ wrkRank ] = index;
            comm.send( wrkRank, 1, data[ index ] );
            ++index;
        }

        virtual void handleResponse( int wrkRank )
        {
            comm.recv( wrkRank, 1, data[ assignedTasks[ wrkRank ] ] );
        }

        virtual void processTask( )
        {
            EOT ind;
            comm.recv( _masterRank, 1, ind );
            func( ind );
            comm.send( _masterRank, 1, ind );
        }

        bool isFinished()
        {
            return index = size;
        }

    protected:
        vector<EOT> & data;
        eoUF<EOT&, void>& func;
        int index;
        int size;
        std::map< int /* worker rank */, int /* index in vector */> assignedTasks;
};

# endif // __EO_PARALLEL_APPLY_H__


