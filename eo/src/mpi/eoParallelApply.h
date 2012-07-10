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
                    std::vector<EOT> & _pop,
                    int _masterRank,
                    // long _maxTime = 0,
                    int _packetSize
                   ) :
                _data( &_pop ), func( _proc ), index( 0 ), size( _pop.size() ), packetSize( _packetSize ), masterRank( _masterRank ), comm( Node::comm() )
            {
                if ( _packetSize <= 0 )
                {
                    throw std::runtime_error("Packet size should not be negative.");
                }
                tempArray = new EOT[ _packetSize ];
            }

            void init( std::vector<EOT>& _pop )
            {
                index = 0;
                size = _pop.size();
                _data = &_pop;
                assignedTasks.clear();
            }

            ~ParallelApplyData()
            {
                delete [] tempArray;
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
            using JobStore< ParallelApplyData<EOT> >::_stf;
            using JobStore< ParallelApplyData<EOT> >::_hrf;
            using JobStore< ParallelApplyData<EOT> >::_ptf;
            using JobStore< ParallelApplyData<EOT> >::_iff;

            ParallelApplyStore(
                    eoUF<EOT&, void> & _proc,
                    std::vector<EOT>& _pop,
                    int _masterRank,
                    // long _maxTime = 0,
                    int _packetSize = 1,
                    // JobStore functors
                    SendTaskParallelApply<EOT> * stpa = new SendTaskParallelApply<EOT>,
                    HandleResponseParallelApply<EOT>* hrpa = new HandleResponseParallelApply<EOT>,
                    ProcessTaskParallelApply<EOT>* ptpa = new ProcessTaskParallelApply<EOT>,
                    IsFinishedParallelApply<EOT>* ifpa = new IsFinishedParallelApply<EOT>
                   ) :
                _data( _proc, _pop, _masterRank, _packetSize )
            {
                _stf = stpa;
                _hrf = hrpa;
                _ptf = ptpa;
                _iff = ifpa;
            }

            ParallelApplyData<EOT>* data() { return &_data; }

            virtual ~ParallelApplyStore()
            {
                delete _stf;
                delete _hrf;
                delete _ptf;
                delete _iff;
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


