# ifndef __EO_PARALLEL_APPLY_H__
# define __EO_PARALLEL_APPLY_H__

# include "eoMpi.h"

# include <eoFunctor.h>
# include <vector>

namespace eo
{
    namespace mpi
    {
        template< typename EOT >
        class ParallelApply : public Job
        {
            private:
                struct ParallelApplyAssignment
                {
                    int index;
                    int size;
                };
            public:

                ParallelApply(
                        eoUF<EOT&, void> & _proc,
                        std::vector<EOT>& _pop,
                        AssignmentAlgorithm & algo,
                        int _masterRank,
                        int _packetSize = 1
                        ) :
                    Job( algo, _masterRank ),
                    func( _proc ),
                    index( 0 ),
                    size( _pop.size() ),
                    data( _pop ),
                    packetSize( _packetSize )
                {
                    if ( _packetSize <= 0 )
                    {
                        throw std::runtime_error("Packet size should not be negative.");
                    }
                    tempArray = new EOT[ packetSize ];
                }

                ~ParallelApply()
                {
                    delete [] tempArray;
                }

                virtual void sendTask( int wrkRank )
                {
                    int futureIndex;

                    if( index + packetSize < size )
                    {
                        futureIndex = index + packetSize;
                    } else {
                        futureIndex = size;
                    }

                    int sentSize = futureIndex - index ;
                    comm.send( wrkRank, 1, sentSize );

                    assignedTasks[ wrkRank ].index = index;
                    assignedTasks[ wrkRank ].size = sentSize;

                    comm.send( wrkRank, 1, &data[ index ] , sentSize );
                    index = futureIndex;
                }

                virtual void handleResponse( int wrkRank )
                {
                    comm.recv( wrkRank, 1, &data[ assignedTasks[wrkRank].index ], assignedTasks[wrkRank].size );
                }

                virtual void processTask( )
                {
                    int recvSize;
                    comm.recv( masterRank, 1, recvSize );
                    comm.recv( masterRank, 1, tempArray, recvSize );
                    for( int i = 0; i < recvSize ; ++i )
                    {
                        func( tempArray[ i ] );
                    }
                    comm.send( masterRank, 1, tempArray, recvSize );
                }

                bool isFinished()
                {
                    return index == size;
                }

            protected:
                std::vector<EOT> & data;
                eoUF<EOT&, void>& func;
                int index;
                int size;
                std::map< int /* worker rank */, ParallelApplyAssignment /* min indexes in vector */> assignedTasks;

                int packetSize;
                EOT* tempArray;
        };
    }
}
# endif // __EO_PARALLEL_APPLY_H__


