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
            ParallelApplyData(
                    eoUF<EOT&, void> & _proc,
                    std::vector<EOT>& _pop,
                    int _masterRank,
                    // long _maxTime = 0,
                    int _packetSize // FIXME = 1 ?
                   ) :
                data( _pop ), func( _proc ), index( 0 ), size( _pop.size() ), packetSize( _packetSize ), masterRank( _masterRank ), comm( Node::comm() )
            {
                if ( _packetSize <= 0 )
                {
                    throw std::runtime_error("Packet size should not be negative.");
                }
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

        template< class EOT >
        class SendTaskParallelApply : public SendTaskFunction< ParallelApplyData<EOT> >
        {
            public:
            using SendTaskFunction< ParallelApplyData<EOT> >::d;

            SendTaskParallelApply( SendTaskParallelApply<EOT> * w = 0 ) : SendTaskFunction< ParallelApplyData<EOT> >( w )
            {
                // empty
            }

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
        class HandleResponseParallelApply : public HandleResponseFunction< ParallelApplyData<EOT> >
        {
            public:
            using HandleResponseFunction< ParallelApplyData<EOT> >::d;

            HandleResponseParallelApply( HandleResponseParallelApply<EOT> * w = 0 ) : HandleResponseFunction< ParallelApplyData<EOT> >( w )
            {
                // empty
            }

            void operator()(int wrkRank)
            {
                d->comm.recv( wrkRank, 1, & (d->data[ d->assignedTasks[wrkRank].index ] ), d->assignedTasks[wrkRank].size );
            }
        };

        template< class EOT >
        class ProcessTaskParallelApply : public ProcessTaskFunction< ParallelApplyData<EOT> >
        {
            public:
            using ProcessTaskFunction< ParallelApplyData<EOT> >::d;

            ProcessTaskParallelApply( ProcessTaskParallelApply<EOT> * w = 0 ) : ProcessTaskFunction< ParallelApplyData<EOT> >( w )
            {
                // empty
            }

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
        class IsFinishedParallelApply : public IsFinishedFunction< ParallelApplyData<EOT> >
        {
            public:
            using IsFinishedFunction< ParallelApplyData<EOT> >::d;

            IsFinishedParallelApply( IsFinishedParallelApply<EOT> * w = 0 ) : IsFinishedFunction< ParallelApplyData<EOT> >( w )
            {
                // empty
            }

            bool operator()()
            {
                return d->index == d->size;
            }
        };

        template< class EOT >
        struct ParallelApplyStore : public JobStore< ParallelApplyData<EOT> >
        {
            ParallelApplyStore(
                    eoUF<EOT&, void> & _proc,
                    std::vector<EOT>& _pop,
                    int _masterRank,
                    // long _maxTime = 0,
                    int _packetSize = 1
                   )
                : _data( _proc, _pop, _masterRank, _packetSize )
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

            SendTaskParallelApply< EOT >& sendTask() const { return *stpa; }
            HandleResponseParallelApply< EOT >& handleResponse() const { return *hrpa; }
            ProcessTaskParallelApply< EOT > & processTask() const { return *ptpa; }
            IsFinishedParallelApply< EOT >& isFinished() const { return *ispa; }

            void sendTask( SendTaskParallelApply< EOT >* _stpa ) { stpa = _stpa; }
            void handleResponse( HandleResponseParallelApply< EOT >* _hrpa ) { hrpa = _hrpa; }
            void processTask( ProcessTaskParallelApply< EOT >* _ptpa ) { ptpa = _ptpa; }
            void isFinished( IsFinishedParallelApply< EOT >* _ispa ) { ispa = _ispa; }

            ParallelApplyData<EOT>* data() { return &_data; }

            protected:
            // TODO commenter : Utiliser des pointeurs pour éviter d'écraser les fonctions wrappées
            SendTaskParallelApply<EOT>* stpa;
            HandleResponseParallelApply<EOT>* hrpa;
            ProcessTaskParallelApply<EOT>* ptpa;
            IsFinishedParallelApply<EOT>* ispa;

            ParallelApplyData<EOT> _data;
        };

        template< typename EOT >
        class ParallelApply : public Job< ParallelApplyData<EOT> >
        {
            public:

            ParallelApply(
                    AssignmentAlgorithm & algo,
                    int _masterRank,
                    ParallelApplyStore<EOT> & store
                    ) :
                Job< ParallelApplyData<EOT> >( algo, _masterRank, store )
            {
                // empty
            }

            protected:

            // bmpi::communicator& comm;
        };
    }
}
# endif // __EO_PARALLEL_APPLY_H__


