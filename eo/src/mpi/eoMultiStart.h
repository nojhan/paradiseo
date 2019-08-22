# ifndef __EO_MULTISTART_H__
# define __EO_MULTISTART_H__

# include "../eo"
# include "eoMpi.h"

/**
 * @ingroup MPI
 * @{
 */

/**
 * @file eoMultiStart.h
 *
 * Contains implementation of a MPI job which consists in a multi start, which basically consists in the following:
 * the same eoAlgo is launched on computers of a clusters, with different seeds for each. As the eoAlgo are most of
 * the time stochastics, the results won't be the same. It is fully equivalent to launch the same program but with
 * different seeds.
 *
 * It follows the structure of a MPI job, as described in eoMpi.h. The basic algorithm is trivial:
 * - Loop while we have a run to perform.
 * - Worker performs runs and send their best solution (individual with best fitness) to the master.
 * - Master retrieves the best solution and adds it to a eoPop of best solutions (the user can chooses what he does
 *   with this population, for instance: retrieve the best element, etc.)
 *
 * The principal concerns about this algorithm are:
 * - How do we reinitialize the algorithm? An eoAlgo can have several forms, and initializations have to be performed
 *   before each "start". We can hence decide whether we reinits the population or keep the same population obtained
 *   after the previous start, we have to reinitialize continuator, etc. This is customizable in the store.
 *
 * - Which seeds should be chosen? If we want the run to be re-runnable with the same results, we need to be sure that
 *   the seeds are the same. But user can not care about this, and just want random seeds. This is customizable in the
 *   store.
 *
 * These concerns are handled by functors, inheriting from MultiStartStore<EOT>::ResetAlgo (for the first concern), and
 * MultiStartStore<EOT>::GetSeeds (for the second one). There are default implementations, but there is no problem about
 * specializing them or coding your own, by directly inheriting from them.
 *
 * @ingroup MPI
 */

namespace eo
{
    namespace mpi
    {
        /**
         * @brief Data used by the Multi Start job.
         *
         * This data is shared between the different Job functors. More details are given for each attribute.
         */
        template< class EOT >
        struct MultiStartData
        {
            typedef eoUF< eoPop<EOT>&, void> ResetAlgo;

            MultiStartData(
                    bmpi::communicator& _comm,
                    eoAlgo<EOT>& _algo,
                    int _masterRank,
                    ResetAlgo & _resetAlgo )
                :
                    runs( 0 ), bests(), pop(),
                    comm( _comm ), algo( _algo ), resetAlgo( _resetAlgo ), masterRank( _masterRank )
            {
                // empty
            }

            // dynamic parameters
            /**
             * @brief Total remaining number of runs.
             *
             * It's decremented as the runs are performed.
             */
            int runs;

            /**
             * @brief eoPop of the best individuals, which are the one sent by the workers.
             */
            eoPop< EOT > bests;

            /**
             * @brief eoPop on which the worker is working.
             */
            eoPop< EOT > pop;

            // static parameters
            /**
             * @brief Communicator, used to send and retrieve messages.
             */
            bmpi::communicator& comm;

            /**
             * @brief Algorithm which will be performed by the worker.
             */
            eoAlgo<EOT>& algo;

            /**
             * @brief Reset Algo functor, which defines how to reset the algo (above) before re running it.
             */
            ResetAlgo& resetAlgo;

            // Rank of master
            int masterRank;
        };

        /**
         * @brief Send task (master side) in the Multi Start job.
         *
         * It only consists in decrementing the number of runs, as the worker already have the population and
         * all the necessary parameters to run the eoAlgo.
         */
        template< class EOT >
        class SendTaskMultiStart : public SendTaskFunction< MultiStartData< EOT > >
        {
            public:
                using SendTaskFunction< MultiStartData< EOT > >::_data;

                void operator()( int wrkRank )
                {
                    wrkRank++; // unused
                    --(_data->runs);
                }
        };

        /**
         * @brief Handle Response (master side) in the Multi Start job.
         *
         * It consists in retrieving the best solution sent by the worker and adds it to a population of best
         * solutions.
         */
        template< class EOT >
        class HandleResponseMultiStart : public HandleResponseFunction< MultiStartData< EOT > >
        {
            public:
                using HandleResponseFunction< MultiStartData< EOT > >::_data;

                void operator()( int wrkRank )
                {
                    EOT individual;
                    MultiStartData< EOT >& d = *_data;
                    d.comm.recv( wrkRank, eo::mpi::Channel::Messages, individual );
                    d.bests.push_back( individual );
                }
        };

        /**
         * @brief Process Task (worker side) in the Multi Start job.
         *
         * Consists in resetting the algorithm and launching it on the population, then
         * send the best individual (the one with the best fitness) to the master.
         */
        template< class EOT >
        class ProcessTaskMultiStart : public ProcessTaskFunction< MultiStartData< EOT > >
        {
            public:
                using ProcessTaskFunction< MultiStartData<EOT > >::_data;

                void operator()()
                {
                    _data->resetAlgo( _data->pop );
                    _data->algo( _data->pop );
                    _data->comm.send( _data->masterRank, eo::mpi::Channel::Messages, _data->pop.best_element() );
                }
        };

        /**
         * @brief Is Finished (master side) in the Multi Start job.
         *
         * The job is finished if and only if all the runs have been performed.
         */
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

        /**
         * @brief Store for the Multi Start job.
         *
         * Contains the data used by the workers (algo,...) and functor to
         * send the seeds.
         */
        template< class EOT >
        class MultiStartStore : public JobStore< MultiStartData< EOT > >
        {
            public:

                /**
                 * @brief Generic functor to reset an algorithm before it's launched by
                 * the worker.
                 *
                 * This reset algorithm should reinits population (if necessary), continuator, etc.
                 */
                typedef typename MultiStartData<EOT>::ResetAlgo ResetAlgo;

                /**
                 * @brief Generic functor which returns a vector of seeds for the workers.
                 *
                 * If this vector hasn't enough seeds to send, random ones are generated and
                 * sent to the workers.
                 */
                typedef eoUF< int, std::vector<int> > GetSeeds;

                /**
                 * @brief Default ctor for MultiStartStore.
                 *
                 * @param algo The algorithm to launch in parallel
                 * @param masterRank The MPI rank of the master
                 * @param resetAlgo The ResetAlgo functor
                 * @param getSeeds The GetSeeds functor
                 */
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
                    // Default job functors for this one.
                    this->_iff = new IsFinishedMultiStart< EOT >;
                    this->_iff->needDelete(true);
                    this->_stf = new SendTaskMultiStart< EOT >;
                    this->_stf->needDelete(true);
                    this->_hrf = new HandleResponseMultiStart< EOT >;
                    this->_hrf->needDelete(true);
                    this->_ptf = new ProcessTaskMultiStart< EOT >;
                    this->_ptf->needDelete(true);
                }

                /**
                 * @brief Send new seeds to the workers before a job.
                 *
                 * Uses the GetSeeds functor given in constructor. If there's not
                 * enough seeds to send, random seeds are sent to the workers.
                 *
                 * @param workers Vector of MPI ranks of the workers
                 * @param runs The number of runs to perform
                 */
                void init( const std::vector<int>& workers, int runs )
                {
                    _data.runs = runs;

                    unsigned nbWorkers = workers.size();
                    std::vector< int > seeds = _getSeeds( nbWorkers );
                    if( eo::mpi::Node::comm().rank() == _masterRank )
                    {
                        if( seeds.size() < nbWorkers )
                        {
                            // Random seeds
                            for( unsigned i = seeds.size(); i < nbWorkers; ++i )
                            {
                                seeds.push_back( eo::rng.rand() );
                            }
                        }

                        for( unsigned i = 0 ; i < nbWorkers ; ++i )
                        {
                            int wrkRank = workers[i];
                            eo::mpi::Node::comm().send( wrkRank, eo::mpi::Channel::Commands, seeds[ i ] );
                        }
                    } else
                    {
                        int seed;
                        eo::mpi::Node::comm().recv( _masterRank, eo::mpi::Channel::Commands, seed );
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

        /**
         * @brief MultiStart job, created for convenience.
         *
         * This is an OneShotJob, which means workers leave it along with
         * the master.
         */
        template< class EOT >
        class MultiStart : public OneShotJob< MultiStartData< EOT > >
        {
            public:

                MultiStart( AssignmentAlgorithm & algo,
                        int masterRank,
                        MultiStartStore< EOT > & store,
                        // dynamic parameters
                        int runs ) :
                    OneShotJob< MultiStartData< EOT > >( algo, masterRank, store )
            {
                store.init( algo.idles(), runs );
            }

            /**
             * @brief Returns the best solution, at the end of the job.
             *
             * Warning: if you call this function from a worker, or from the master before the
             * launch of the job, you will only get an empty population!
             *
             * @return Population of best individuals retrieved by the master.
             */
            eoPop<EOT>& best_individuals()
            {
                return this->store.data()->bests;
            }
        };

        /*************************************
         * DEFAULT GET SEEDS IMPLEMENTATIONS *
         ************************************/

        /**
         * @brief Uses the internal default seed generator to get seeds,
         * which means: random seeds are sent.
         */
        template<class EOT>
        struct DummyGetSeeds : public MultiStartStore<EOT>::GetSeeds
        {
            std::vector<int> operator()( int n )
            {
                return std::vector<int>();
            }
        };

        /**
         * @brief Sends seeds to the workers, which are multiple of a number
         * given by the master. If no number is given, a random one is used.
         *
         * This functor ensures that even if the same store is used with
         * different jobs, the seeds will be different.
         */
        template<class EOT>
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

        /**
         * @brief Returns random seeds to the workers. We can controle which seeds are generated
         * by precising the seed of the master.
         */
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

        /**************************************
         * DEFAULT RESET ALGO IMPLEMENTATIONS *
         **************************************/

        /**
         * @brief For a Genetic Algorithm, reinits the population by copying the original one
         * given in constructor, and reinits the continuator.
         *
         * The evaluator should also be given, as the population needs to be evaluated
         * before each run.
         */
        template<class EOT>
        struct ReuseOriginalPopEA: public MultiStartStore<EOT>::ResetAlgo
        {
            ReuseOriginalPopEA(
                    eoCountContinue<EOT> & continuator,
                    const eoPop<EOT>& originalPop,
                    eoEvalFunc<EOT>& eval) :
                _continuator( continuator ),
                _originalPop( originalPop ),
                _pop_eval( eval )
            {
                // empty
            }

            ReuseOriginalPopEA(
                    eoCountContinue<EOT> & continuator,
                    const eoPop<EOT>& originalPop,
                    eoPopEvalFunc<EOT>& pop_eval
                    ) :
                _continuator( continuator ),
                _originalPop( originalPop ),
                _pop_eval( pop_eval )
            {
                // empty
            }

            void operator()( eoPop<EOT>& pop )
            {
                pop = _originalPop; // copies the original population
                _pop_eval( pop, pop );
                _continuator.reset();
            }

            private:
            eoCountContinue<EOT> & _continuator;
            const eoPop<EOT>& _originalPop;
            eoPopEvalFunc<EOT>& _pop_eval;
        };

        /**
         * @brief For a Genetic Algorithm, reuses the same population without
         * modifying it after a run.
         *
         * This means, if you launch a run after another one, you'll make evolve
         * the same population.
         *
         * The evaluator should also be sent, as the population needs to be evaluated
         * at the first time.
         */
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

            ReuseSamePopEA(
                    eoCountContinue<EOT>& continuator,
                    const eoPop<EOT>& originalPop,
                    eoPopEvalFunc<EOT>& pop_eval
                    ) :
                _continuator( continuator ),
                _originalPop( originalPop ),
                _firstTime( true )
            {
                pop_eval( _originalPop, _originalPop );
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
    } // namespace mpi
} // namespace eo

/**
 * @}
 */

# endif // __EO_MULTISTART_H__
