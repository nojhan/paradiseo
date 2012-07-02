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
        class SendTaskParallelApply;

        template<class EOT>
        class HandleResponseParallelApply;

        template<class EOT>
        class ProcessTaskParallelApply;

        template<class EOT>
        class IsFinishedParallelApply;

        template<class EOT>
        class ParallelApply;

        template< class EOT >
        class BaseParallelApply
        {
            public:
            void owner(ParallelApply<EOT> * job)
            {
                j = job;
            }

            protected:
            ParallelApply<EOT> * j;
        };

        template< typename EOT >
        class ParallelApply : public Job
        {
            friend class SendTaskParallelApply<EOT>;
            friend class HandleResponseParallelApply<EOT>;
            friend class ProcessTaskParallelApply<EOT>;
            friend class IsFinishedParallelApply<EOT>;

            public:

            ParallelApply(
                    eoUF<EOT&, void> & _proc,
                    std::vector<EOT>& _pop,
                    AssignmentAlgorithm & algo,
                    int _masterRank,
                    const JobStore& store,
                    // long _maxTime = 0,
                    int _packetSize = 1
                    ) :
                Job( algo, _masterRank, store ),
                // Job( algo, _masterRank, _maxTime ),
                func( _proc ),
                data( _pop ),
                packetSize( _packetSize ),
                index( 0 ),
                size( _pop.size() )
            {
                if ( _packetSize <= 0 )
                {
                    throw std::runtime_error("Packet size should not be negative.");
                }
                tempArray = new EOT [ _packetSize ];

                dynamic_cast< BaseParallelApply<EOT>& >( sendTask ).owner( this );
                dynamic_cast< BaseParallelApply<EOT>& >( handleResponse ).owner( this );
                dynamic_cast< BaseParallelApply<EOT>& >( processTask ).owner( this );
                dynamic_cast< BaseParallelApply<EOT>& >( isFinished ).owner( this );
            }

            ~ParallelApply()
            {
                delete [] tempArray;
            }

            protected:

            std::vector<EOT> & data;
            eoUF<EOT&, void> & func;
            int index;
            int size;
            std::map< int /* worker rank */, ParallelApplyAssignment /* min indexes in vector */> assignedTasks;
            int packetSize;
            EOT* tempArray;

            // bmpi::communicator& comm;
        };

        template< class EOT >
        class SendTaskParallelApply : public SendTaskFunction, public BaseParallelApply<EOT>
        {
            public:
            using BaseParallelApply<EOT>::j;

            // futureIndex, index, packetSize, size, comm, assignedTasks, data
            void operator()(int wrkRank)
            {
                int futureIndex;

                if( j->index + j->packetSize < j->size )
                {
                    futureIndex = j->index + j->packetSize;
                } else {
                    futureIndex = j->size;
                }

                int sentSize = futureIndex - j->index ;

                j->comm.send( wrkRank, 1, sentSize );

                eo::log << eo::progress << "Evaluating individual " << j->index << std::endl;

                j->assignedTasks[ wrkRank ].index = j->index;
                j->assignedTasks[ wrkRank ].size = sentSize;

                j->comm.send( wrkRank, 1, & ( (j->data)[ j->index ] ) , sentSize );
                j->index = futureIndex;
            }
        };

        template< class EOT >
        class HandleResponseParallelApply : public HandleResponseFunction, public BaseParallelApply<EOT>
        {
            public:
            using BaseParallelApply<EOT>::j;

            void operator()(int wrkRank)
            {
                j->comm.recv( wrkRank, 1, & (j->data[ j->assignedTasks[wrkRank].index ] ), j->assignedTasks[wrkRank].size );
            }
        };

        template< class EOT >
        class ProcessTaskParallelApply : public ProcessTaskFunction, public BaseParallelApply<EOT>
        {
            public:
            using BaseParallelApply<EOT>::j;

            void operator()()
            {
                int recvSize;

                j->comm.recv( j->masterRank, 1, recvSize );
                j->comm.recv( j->masterRank, 1, j->tempArray, recvSize );
                timerStat.start("worker_processes");
                for( int i = 0; i < recvSize ; ++i )
                {
                    j->func( j->tempArray[ i ] );
                }
                timerStat.stop("worker_processes");
                j->comm.send( j->masterRank, 1, j->tempArray, recvSize );
            }
        };

        template< class EOT >
        class IsFinishedParallelApply : public IsFinishedFunction, public BaseParallelApply<EOT>
        {
            public:

            using BaseParallelApply<EOT>::j;

            bool operator()()
            {
                return j->index == j->size;
            }
        };

        template< class EOT >
        struct ParallelApplyStore : public JobStore
        {
            ParallelApplyStore()
            {
                stpa = new SendTaskParallelApply<EOT>;
                hrpa = new HandleResponseParallelApply<EOT>;
                ptpa = new ProcessTaskParallelApply<EOT>;
                ispa = new IsFinishedParallelApply<EOT>;
            }

            ~ParallelApplyStore()
            {
                delete stpa;
                delete hrpa;
                delete ptpa;
                delete ispa;
            }

            SendTaskFunction& sendTask() const { return *stpa; }
            HandleResponseFunction& handleResponse() const { return *hrpa; }
            ProcessTaskFunction& processTask() const { return *ptpa; }
            IsFinishedFunction& isFinished() const { return *ispa; }

            protected:
            SendTaskParallelApply<EOT>* stpa;
            HandleResponseParallelApply<EOT>* hrpa;
            ProcessTaskParallelApply<EOT>* ptpa;
            IsFinishedParallelApply<EOT>* ispa;
        };
    }
}
# endif // __EO_PARALLEL_APPLY_H__


