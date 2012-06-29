# ifndef __EO_PARALLEL_APPLY_H__
# define __EO_PARALLEL_APPLY_H__

# include "eoMpi.h"

# include <eoFunctor.h>
# include <vector>

namespace eo
{
    namespace mpi
    {
        struct ParallelApplyAssignment
        {
            int index;
            int size;
        };

        template<class EOT>
        struct ParallelApplyData
        {
            ParallelApplyData() {}

            ParallelApplyData(
                    eoUF<EOT&, void> & _proc,
                    std::vector<EOT>& _pop,
                    int _masterRank,
                    int _packetSize
                    ) :
                func( _proc ),
                data( _pop ),
                index( 0 ),
                size( _pop.size() ),
                packetSize( _packetSize ),
                // job
                masterRank( _masterRank ),
                comm( Node::comm() )
            {
                tempArray = new EOT[ _packetSize ];
            }

            ~ParallelApplyData()
             {
                 delete [] tempArray;
             }

            std::vector<EOT> & data;
            eoUF<EOT&, void> & func;
            int index;
            int size;
            std::map< int /* worker rank */, ParallelApplyAssignment /* min indexes in vector */> assignedTasks;

            int packetSize;
            EOT* tempArray;

            int masterRank;
            bmpi::communicator& comm;
        };

        template<class Data>
        struct SendTaskParallelApply : public SendTaskFunction< Data >
        {
            SendTaskParallelApply( Data & _d )
            {
                data( _d );
            }

            using SharedDataFunction< Data >::d;

            // futureIndex, index, packetSize, size, comm, assignedTasks, data
            void operator()(int wrkRank)
            {
                int futureIndex;

                if( d.index + d.packetSize < d.size )
                {
                    futureIndex = d.index + d.packetSize;
                } else {
                    futureIndex = d.size;
                }

                int sentSize = futureIndex - d.index ;

                d.comm.send( wrkRank, 1, sentSize );

                eo::log << eo::progress << "Evaluating individual " << d.index << std::endl;

                d.assignedTasks[ wrkRank ].index = d.index;
                d.assignedTasks[ wrkRank ].size = sentSize;

                d.comm.send( wrkRank, 1, & (d.data[ index ]) , sentSize );
                d.index = futureIndex;
            }
        };

        template<class Data>
        struct HandleResponseParallelApply : public HandleResponseFunction< Data >
        {
            HandleResponseParallelApply( Data & _d )
            {
                data( _d );
            }

            using SharedDataFunction< Data >::d;
            void operator()(int wrkRank)
            {
                d.comm.recv( wrkRank, 1, & (d.data[ d.assignedTasks[wrkRank].index ] ), d.assignedTasks[wrkRank].size );
            }
        };

        template<class Data>
        struct ProcessTaskParallelApply : public ProcessTaskFunction< Data >
        {
            ProcessTaskParallelApply( Data & _d )
            {
                data( _d );
            }

            using SharedDataFunction< Data >::d;
            void operator()()
            {
                int recvSize;
                d.comm.recv( d.masterRank, 1, recvSize );
                d.comm.recv( d.masterRank, 1, d.tempArray, recvSize );
                timerStat.start("worker_processes");
                for( int i = 0; i < recvSize ; ++i )
                {
                    d.func( d.tempArray[ i ] );
                }
                timerStat.stop("worker_processes");
                d.comm.send( d.masterRank, 1, d.tempArray, recvSize );
            }
        };

        template<class Data>
        struct IsFinishedParallelApply : public IsFinishedFunction< Data >
        {
            IsFinishedParallelApply( Data & _d )
            {
                data( _d );
            }

            using SharedDataFunction< Data >::d;
            bool operator()()
            {
                return d.index == d.size;
            }
        };

        template< typename Data >
        struct ParallelApplyStore : public JobStore< Data >
        {
            ParallelApplyStore( Data & data )
            {
                stpa = new SendTaskParallelApply< Data >( data );
                hrpa = new HandleResponseParallelApply< Data >( data );
                ptpa = new ProcessTaskParallelApply< Data >( data );
                ispa = new IsFinishedParallelApply< Data >( data );
            }

            ~ParallelApplyStore()
            {
                delete stpa;
                delete hrpa;
                delete ptpa;
                delete ispa;
            }

            SendTaskFunction< Data > & sendTask() { return *stpa; }
            HandleResponseFunction< Data > & handleResponse() { return *hrpa; }
            ProcessTaskFunction< Data > & processTask() { return *ptpa; }
            IsFinishedFunction< Data > & isFinished() { return *ispa; }

            protected:
            SendTaskParallelApply< Data >* stpa;
            HandleResponseParallelApply< Data >* hrpa;
            ProcessTaskParallelApply< Data >* ptpa;
            IsFinishedParallelApply< Data >* ispa;
        };

        template< typename EOT >
            class ParallelApply : public Job< ParallelApplyData<EOT> >
        {
            public:

                ParallelApply(
                        eoUF<EOT&, void> & _proc,
                        std::vector<EOT>& _pop,
                        AssignmentAlgorithm & algo,
                        int _masterRank,
                        // long _maxTime = 0,
                        int _packetSize = 1
                        ) :

                    Job< ParallelApplyData<EOT> >( algo, _masterRank, ParallelApplyStore< ParallelApplyData<EOT> >( sharedData ) ),
                    // Job( algo, _masterRank, _maxTime ),
                    sharedData( _proc, _pop, _masterRank, _packetSize )

            {
                if ( _packetSize <= 0 )
                {
                    throw std::runtime_error("Packet size should not be negative.");
                }
            }

            protected:
                ParallelApplyData<EOT> sharedData;
        };
    }
}
# endif // __EO_PARALLEL_APPLY_H__


