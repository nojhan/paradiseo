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
        struct JobData
        {
            JobData(
                    eoUF<EOT&, void> & _proc,
                    std::vector<EOT>& _pop,
                    int _masterRank,
                    // long _maxTime = 0,
                    int _packetSize
                   ) :
                data( _pop ), func( _proc ), index( 0 ), size( _pop.size() ), packetSize( _packetSize ), masterRank( _masterRank ), comm( Node::comm() )
            {
                if ( _packetSize <= 0 )
                {
                    throw std::runtime_error("Packet size should not be negative.");
                }
                tempArray = new EOT[ _packetSize ];
            }

            ~JobData()
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

        /*
        template< typename EOT >
        class ParallelApply : public Job< JobData<EOT> >
        {
            public:

            ParallelApply(
                    // eoUF<EOT&, void> & _proc,
                    // std::vector<EOT>& _pop,
                    AssignmentAlgorithm & algo,
                    int _masterRank,
                    const JobStore< JobData<EOT> >& store
                    // long _maxTime = 0,
                    // int _packetSize = 1
                    ) :
                Job( algo, _masterRank, store )
                // Job( algo, _masterRank, _maxTime ),
                func( _proc ),
                data( _pop ),
                packetSize( _packetSize ),
                index( 0 ),
                size( _pop.size() )
            {
                // empty
            }
        
            protected:

            // bmpi::communicator& comm;
        };
        */

        template< class EOT >
        class SendTaskParallelApply : public SendTaskFunction< JobData<EOT> >
        {
            public:
            using SendTaskFunction< JobData<EOT> >::d;

            // futureIndex, index, packetSize, size, comm, assignedTasks, data
            void operator()(int wrkRank)
            {
                int futureIndex;

                if( d->index + d->packetSize < d->size )
                {
                    futureIndex = d->index + d->packetSize;
                } else {
                    futureIndex = d->size;
                }

                int sentSize = futureIndex - d->index ;

                d->comm.send( wrkRank, 1, sentSize );

                eo::log << eo::progress << "Evaluating individual " << d->index << std::endl;

                d->assignedTasks[ wrkRank ].index = d->index;
                d->assignedTasks[ wrkRank ].size = sentSize;

                d->comm.send( wrkRank, 1, & ( (d->data)[ d->index ] ) , sentSize );
                d->index = futureIndex;
            }
        };

        template< class EOT >
        class HandleResponseParallelApply : public HandleResponseFunction< JobData<EOT> >
        {
            public:
            using HandleResponseFunction< JobData<EOT> >::d;

            void operator()(int wrkRank)
            {
                d->comm.recv( wrkRank, 1, & (d->data[ d->assignedTasks[wrkRank].index ] ), d->assignedTasks[wrkRank].size );
            }
        };

        template< class EOT >
        class ProcessTaskParallelApply : public ProcessTaskFunction< JobData<EOT> >
        {
            public:
            using ProcessTaskFunction< JobData<EOT> >::d;

            void operator()()
            {
                int recvSize;

                d->comm.recv( d->masterRank, 1, recvSize );
                d->comm.recv( d->masterRank, 1, d->tempArray, recvSize );
                timerStat.start("worker_processes");
                for( int i = 0; i < recvSize ; ++i )
                {
                    d->func( d->tempArray[ i ] );
                }
                timerStat.stop("worker_processes");
                d->comm.send( d->masterRank, 1, d->tempArray, recvSize );
            }
        };

        template< class EOT >
        class IsFinishedParallelApply : public IsFinishedFunction< JobData<EOT> >
        {
            public:
            using IsFinishedFunction< JobData<EOT> >::d;

            bool operator()()
            {
                return d->index == d->size;
            }
        };

        template< class EOT >
        struct ParallelApplyStore : public JobStore< JobData<EOT> >
        {
            ParallelApplyStore(
                    eoUF<EOT&, void> & _proc,
                    std::vector<EOT>& _pop,
                    int _masterRank,
                    // long _maxTime = 0,
                    int _packetSize = 1
                   )
                : j( _proc, _pop, _masterRank, _packetSize )
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

            SendTaskFunction< JobData<EOT> >& sendTask() const { return *stpa; }
            HandleResponseFunction< JobData<EOT> >& handleResponse() const { return *hrpa; }
            ProcessTaskFunction< JobData<EOT> >& processTask() const { return *ptpa; }
            IsFinishedFunction< JobData<EOT> >& isFinished() const { return *ispa; }

            JobData<EOT>* data() { return &j; }

            protected:
            SendTaskParallelApply<EOT>* stpa;
            HandleResponseParallelApply<EOT>* hrpa;
            ProcessTaskParallelApply<EOT>* ptpa;
            IsFinishedParallelApply<EOT>* ispa;

            JobData<EOT> j;
        };
    }
}
# endif // __EO_PARALLEL_APPLY_H__


