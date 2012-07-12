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
                    int _masterRank,
                    // long _maxTime = 0,
                    int _packetSize,
                    std::vector<EOT> * _pop = 0
                   ) :
                _data( _pop ), func( _proc ), index( 0 ), packetSize( _packetSize ), masterRank( _masterRank ), comm( Node::comm() )
            {
                if ( _packetSize <= 0 )
                {
                    throw std::runtime_error("Packet size should not be negative.");
                }

                if( _pop )
                {
                    size = _pop->size();
                }
            }

            void init( std::vector<EOT>& _pop )
            {
                index = 0;
                size = _pop.size();
                _data = &_pop;
                assignedTasks.clear();
            }

            std::vector<EOT>& data()
            {
                return *_data;
            }

            std::vector<EOT> * _data;
            eoUF<EOT&, void> & func;
            int index;
            int size;
            std::map< int /* worker rank */, ParallelApplyAssignment /* min indexes in vector */> assignedTasks;
            int packetSize;
            std::vector<EOT> tempArray;

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

                d->comm.send( wrkRank, 1, & ( (d->data())[ d->index ] ) , sentSize );
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
                d->comm.recv( wrkRank, 1, & (d->data()[ d->assignedTasks[wrkRank].index ] ), d->assignedTasks[wrkRank].size );
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
                d->tempArray.resize( recvSize );
                d->comm.recv( d->masterRank, 1, & d->tempArray[0] , recvSize );
                timerStat.start("worker_processes");
                for( int i = 0; i < recvSize ; ++i )
                {
                    d->func( d->tempArray[ i ] );
                }
                timerStat.stop("worker_processes");
                d->comm.send( d->masterRank, 1, & d->tempArray[0], recvSize );
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
            using JobStore< ParallelApplyData<EOT> >::_stf;
            using JobStore< ParallelApplyData<EOT> >::_hrf;
            using JobStore< ParallelApplyData<EOT> >::_ptf;
            using JobStore< ParallelApplyData<EOT> >::_iff;

            ParallelApplyStore(
                    eoUF<EOT&, void> & _proc,
                    int _masterRank,
                    int _packetSize = 1,
                    // JobStore functors
                    SendTaskParallelApply<EOT> * stpa = 0,
                    HandleResponseParallelApply<EOT>* hrpa = 0,
                    ProcessTaskParallelApply<EOT>* ptpa = 0,
                    IsFinishedParallelApply<EOT>* ifpa = 0
                   ) :
                _data( _proc, _masterRank, _packetSize )
            {
                if( stpa == 0 ) {
                    stpa = new SendTaskParallelApply<EOT>;
                    stpa->needDelete( true );
                }

                if( hrpa == 0 ) {
                    hrpa = new HandleResponseParallelApply<EOT>;
                    hrpa->needDelete( true );
                }

                if( ptpa == 0 ) {
                    ptpa = new ProcessTaskParallelApply<EOT>;
                    ptpa->needDelete( true );
                }

                if( ifpa == 0 ) {
                    ifpa = new IsFinishedParallelApply<EOT>;
                    ifpa->needDelete( true );
                }

                _stf = stpa;
                _hrf = hrpa;
                _ptf = ptpa;
                _iff = ifpa;
            }

            ParallelApplyData<EOT>* data() { return &_data; }

            void data( std::vector<EOT>& _pop )
            {
                _data.init( _pop );
            }

            virtual ~ParallelApplyStore()
            {
            }

            protected:
            ParallelApplyData<EOT> _data;
        };

        // TODO commentaire : impossible de faire un typedef sur un template sans passer
        // par un traits => complique la tâche de l'utilisateur pour rien.
        template< typename EOT >
        class ParallelApply : public MultiJob< ParallelApplyData<EOT> >
        {
            public:

            ParallelApply(
                    AssignmentAlgorithm & algo,
                    int _masterRank,
                    ParallelApplyStore<EOT> & store
                    ) :
                MultiJob< ParallelApplyData<EOT> >( algo, _masterRank, store )
            {
                // empty
            }
        };
    }
}
# endif // __EO_PARALLEL_APPLY_H__


