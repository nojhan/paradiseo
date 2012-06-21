# ifndef __EO_PARALLEL_APPLY_H__
# define __EO_PARALLEL_APPLY_H__

# include "eompi.h"

# include <eoFunctor.h>
# include <vector>

template< typename EOT >
struct ParallelApplyContinuator : public BaseContinuator
{
    ParallelApplyContinuator( int index, int size )
    {
        _index = index;
        _size = size;
    }

    void index( int i ) { _index = i; }

    bool operator()()
    {
        return _index < _size;
    }

private:
    int _index;
    int _size;
};

template< typename EOT >
class ParallelApply : public MpiJob
{
    public:

        ParallelApply( eoUF<EOT&, void> & _proc, std::vector<EOT>& _pop, AssignmentAlgorithm & algo, int _masterRank ) :
            MpiJob( algo,
                    new ParallelApplyContinuator<EOT>( 0, _pop.size() ),
                    _masterRank ),
            func( _proc ),
            index( 0 ),
            data( _pop )
        {
            pa_continuator = static_cast<ParallelApplyContinuator<EOT>*>( _continuator );
        }

        virtual void sendTask( int wrkRank )
        {
            assignedTasks[ wrkRank ] = index;
            comm.send( wrkRank, 1, data[ index ] );
            ++index;
            pa_continuator->index( index );
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

    protected:
        vector<EOT> & data;
        eoUF<EOT&, void>& func;
        int index;
        ParallelApplyContinuator<EOT> * pa_continuator;
        std::map< int /* worker rank */, int /* index in vector */> assignedTasks;
};

# endif // __EO_PARALLEL_APPLY_H__


