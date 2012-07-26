# ifndef __EO_MULTISTART_H__
# define __EO_MULTISTART_H__

# include <eo>
# include "eoMpi.h"

namespace eo
{
    namespace mpi
    {
        template< class EOT >
        struct MultiStartData
        {
            typedef eoUF< eoPop<EOT>&, void> ResetAlgo;

            MultiStartData( bmpi::communicator& _comm, eoAlgo<EOT>& _algo, int _masterRank, ResetAlgo & _resetAlgo )
                :
                    runs( 0 ), pop(), bests(),
                    comm( _comm ), algo( _algo ), masterRank( _masterRank ), resetAlgo( _resetAlgo )
            {
                // empty
            }

            // dynamic parameters
            int runs;
            eoPop< EOT > bests;
            eoPop< EOT > pop;

            // static parameters
            bmpi::communicator& comm;
            eoAlgo<EOT>& algo;
            ResetAlgo& resetAlgo;
            int masterRank;
        };

        template< class EOT >
        class SendTaskMultiStart : public SendTaskFunction< MultiStartData< EOT > >
        {
            public:
                using SendTaskFunction< MultiStartData< EOT > >::_data;

                void operator()( int wrkRank )
                {
                    --(_data->runs);
                }
        };

        template< class EOT >
        class HandleResponseMultiStart : public HandleResponseFunction< MultiStartData< EOT > >
        {
            public:
                using HandleResponseFunction< MultiStartData< EOT > >::_data;

                void operator()( int wrkRank )
                {
                    EOT individual;
                    MultiStartData< EOT >& d = *_data;
                    d.comm.recv( wrkRank, 1, individual );
                    d.bests.push_back( individual );
                }
        };

        template< class EOT >
        class ProcessTaskMultiStart : public ProcessTaskFunction< MultiStartData< EOT > >
        {
            public:
                using ProcessTaskFunction< MultiStartData<EOT > >::_data;

                void operator()()
                {
                    _data->resetAlgo( _data->pop );
                    _data->algo( _data->pop );
                    _data->comm.send( _data->masterRank, 1, _data->pop.best_element() );
                }
        };

        template< class EOT >
        class IsFinishedMultiStart : public IsFinishedFunction< MultiStartData< EOT > >
        {
            public:
                using IsFinishedFunction< MultiStartData< EOT > >::_data;

                bool operator()()
                {
                    return _data->runs <= 0;
                }
        };

        template< class EOT >
        class MultiStartStore : public JobStore< MultiStartData< EOT > >
        {
            public:

                typedef typename MultiStartData<EOT>::ResetAlgo ResetAlgo;
                typedef eoUF< int, std::vector<int> > GetSeeds;

                MultiStartStore(
                        eoAlgo<EOT> & algo,
                        int masterRank,
                        ResetAlgo & resetAlgo,
                        GetSeeds & getSeeds
                        )
                    : _data( eo::mpi::Node::comm(), algo, masterRank, resetAlgo ),
                    _getSeeds( getSeeds ),
                    _masterRank( masterRank )
            {
                this->_iff = new IsFinishedMultiStart< EOT >;
                this->_iff->needDelete(true);
                this->_stf = new SendTaskMultiStart< EOT >;
                this->_stf->needDelete(true);
                this->_hrf = new HandleResponseMultiStart< EOT >;
                this->_hrf->needDelete(true);
                this->_ptf = new ProcessTaskMultiStart< EOT >;
                this->_ptf->needDelete(true);
            }

                void init( const std::vector<int>& workers, int runs )
                {
                    _data.runs = runs;

                    int nbWorkers = workers.size();
                    std::vector< int > seeds = _getSeeds( nbWorkers );
                    if( eo::mpi::Node::comm().rank() == _masterRank )
                    {
                        if( seeds.size() < nbWorkers )
                        {
                            // Random seeds
                            for( int i = seeds.size(); i < nbWorkers; ++i )
                            {
                                seeds.push_back( eo::rng.rand() );
                            }
                        }

                        for( int i = 0 ; i < nbWorkers ; ++i )
                        {
                            int wrkRank = workers[i];
                            eo::mpi::Node::comm().send( wrkRank, 1, seeds[ i ] );
                        }
                    } else
                    {
                        int seed;
                        eo::mpi::Node::comm().recv( _masterRank, 1, seed );
                        eo::log << eo::debug << eo::mpi::Node::comm().rank() << "- Seed: " << seed << std::endl;
                        eo::rng.reseed( seed );
                    }
                }

                MultiStartData<EOT>* data()
                {
                    return &_data;
                }

            private:
                MultiStartData< EOT > _data;
                GetSeeds & _getSeeds;
                int _masterRank;
        };

        template<class EOT>
        // No seeds! Use default generator
        struct DummyGetSeeds : public MultiStartStore<EOT>::GetSeeds
        {
            std::vector<int> operator()( int n )
            {
                return std::vector<int>();
            }
        };

        template<class EOT>
        // Multiple of a seed
        struct MultiplesOfNumber : public MultiStartStore<EOT>::GetSeeds
        {
            MultiplesOfNumber ( int n = 0 )
            {
                while( n == 0 )
                {
                    n = eo::rng.rand();
                }
                _seed = n;
                _i = 0;
            }

            std::vector<int> operator()( int n )
            {
                std::vector<int> ret;
                for( unsigned int i = 0; i < n; ++i )
                {
                    ret.push_back( (++_i) * _seed );
                }
                return ret;
            }

            private:

            unsigned int _seed;
            unsigned int _i;
        };

        template<class EOT>
        struct GetRandomSeeds : public MultiStartStore<EOT>::GetSeeds
        {
            GetRandomSeeds( int seed )
            {
                eo::rng.reseed( seed );
            }

            std::vector<int> operator()( int n )
            {
                std::vector<int> ret;
                for(int i = 0; i < n; ++i)
                {
                    ret.push_back( eo::rng.rand() );
                }
                return ret;
            }
        };

        template<class EOT>
        struct ReuseOriginalPopEA: public MultiStartStore<EOT>::ResetAlgo
        {
            ReuseOriginalPopEA(
                    eoCountContinue<EOT> & continuator,
                    const eoPop<EOT>& originalPop,
                    eoEvalFunc<EOT>& eval) :
                _continuator( continuator ),
                _originalPop( originalPop ),
                _eval( eval )
            {
                // empty
            }

            void operator()( eoPop<EOT>& pop )
            {
                pop = _originalPop;
                for(unsigned i = 0, size = pop.size(); i < size; ++i)
                {
                    _eval( pop[i] );
                }
                _continuator.reset();
            }

            private:
            eoCountContinue<EOT> & _continuator;
            const eoPop<EOT>& _originalPop;
            eoEvalFunc<EOT>& _eval;
        };

        template< class EOT >
        struct ReuseSamePopEA : public MultiStartStore<EOT>::ResetAlgo
        {
            ReuseSamePopEA(
                    eoCountContinue<EOT>& continuator,
                    const eoPop<EOT>& originalPop,
                    eoEvalFunc<EOT>& eval
                    ) :
                _continuator( continuator ),
                _originalPop( originalPop ),
                _firstTime( true )
            {
                for( unsigned i = 0, size = originalPop.size();
                        i < size; ++i )
                {
                    eval(_originalPop[i]);
                }
            }

            void operator()( eoPop<EOT>& pop )
            {
                if( _firstTime )
                {
                    pop = _originalPop;
                    _firstTime = false;
                }
                _continuator.reset();
            }

            protected:

            eoCountContinue<EOT>& _continuator;
            eoPop<EOT> _originalPop;
            bool _firstTime;
        };

        template< class EOT >
        class MultiStart : public OneShotJob< MultiStartData< EOT > >
        {
            public:

                MultiStart( AssignmentAlgorithm & algo,
                        int masterRank,
                        MultiStartStore< EOT > & store,
                        // dynamic parameters
                        int runs,
                        const std::vector<int>& seeds = std::vector<int>()  ) :
                    OneShotJob< MultiStartData< EOT > >( algo, masterRank, store )
            {
                store.init( algo.idles(), runs );
            }

                eoPop<EOT>& best_individuals()
                {
                    return this->store.data()->bests;
                }
        };

    } // namespace mpi

} // namespace eo

# endif // __EO_MULTISTART_H__
