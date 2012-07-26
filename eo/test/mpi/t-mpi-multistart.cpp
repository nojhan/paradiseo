# include <mpi/eoMpi.h>
using namespace eo::mpi;

#include <stdexcept>
#include <iostream>
#include <sstream>

#include <eo>
#include <es.h>

// Use functions from namespace std
using namespace std;


class SerializableEOReal: public eoReal<double>, public eoserial::Persistent
{
public:

    SerializableEOReal(unsigned size = 0, double value = 0.0) :
            eoReal<double>(size, value)
    {
    }

    void unpack( const eoserial::Object* obj )
    {
        this->clear();
        eoserial::unpackArray
            < std::vector<double>, eoserial::Array::UnpackAlgorithm >
            ( *obj, "vector", *this );

        bool invalidFitness;
        eoserial::unpack( *obj, "invalid_fitness", invalidFitness );
        if( invalidFitness )
        {
            this->invalidate();
        } else
        {
            double f;
            eoserial::unpack( *obj, "fitness", f );
            this->fitness( f );
        }
    }

    eoserial::Object* pack( void ) const
    {
        eoserial::Object* obj = new eoserial::Object;
        obj->add( "vector", eoserial::makeArray< std::vector<double>, eoserial::MakeAlgorithm >( *this ) );

        bool invalidFitness = this->invalid();
        obj->add( "invalid_fitness", eoserial::make( invalidFitness ) );
        if( !invalidFitness )
        {
            obj->add( "fitness", eoserial::make( this->fitness() ) );
        }

        return obj;
    }
};

// REPRESENTATION
//-----------------------------------------------------------------------------
// define your individuals
typedef SerializableEOReal Indi;

// EVAL
//-----------------------------------------------------------------------------
// a simple fitness function that computes the euclidian norm of a real vector
//    @param _indi A real-valued individual

double real_value(const Indi & _indi)
{
  double sum = 0;
  for (unsigned i = 0; i < _indi.size(); i++)
      sum += _indi[i]*_indi[i];
  return (-sum);            // maximizing only
}

/************************** PARALLELIZATION JOB *******************************/
/*
 * This file is a template for a new eo::mpi::Job. You have everything that should be necessary to implement a new
 * parallelized algorithm.
 *
 * Replace MultiStart by the name of your algorithm (for instance: MultiStart, ParallelApply, etc.).
 */

template< class T >
struct SerializableBasicType : public eoserial::Persistent
{
    public:
        SerializableBasicType( T & value )
        {
            _value = value;
        }

        operator T&()
        {
            return _value;
        }

        void unpack( const eoserial::Object* obj )
        {
            eoserial::unpack( *obj, "value", _value );
        }

        eoserial::Object* pack( void ) const
        {
            eoserial::Object* obj = new eoserial::Object;
            obj->add( "value", eoserial::make( _value ) );
        }

    private:
        T _value;
};

template< class EOT >
struct MultiStartData
{
    typedef eoF<void> ResetAlgo;

    MultiStartData( mpi::communicator& _comm, eoAlgo<EOT>& _algo, int _masterRank, ResetAlgo & _resetAlgo )
        :
        runs( 0 ), firstIndividual( true ), bestFitness(), pop(),
        comm( _comm ), algo( _algo ), masterRank( _masterRank ), resetAlgo( _resetAlgo )
    {
        // empty
    }

    // dynamic parameters
    int runs;
    bool firstIndividual;
    typename EOT::Fitness bestFitness;
    EOT bestIndividual;
    eoPop< EOT > pop;

    // static parameters
    mpi::communicator& comm;
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
            std::cout << "Response received." << std::endl;

            EOT individual;
            MultiStartData< EOT >& d = *_data;
            d.comm.recv( wrkRank, 1, individual );

            std::cout << "Fitness of individual: " << individual.fitness() << std::endl;
            if ( ! d.firstIndividual ) {
                std::cout << "Best fitness: " << d.bestFitness << std::endl;
            }

            if( d.firstIndividual || individual.fitness() > d.bestFitness )
            {
                d.bestFitness = individual.fitness();
                d.bestIndividual = individual;
                d.firstIndividual = false;
            }
        }
};

template< class EOT >
class ProcessTaskMultiStart : public ProcessTaskFunction< MultiStartData< EOT > >
{
    public:
        using ProcessTaskFunction< MultiStartData<EOT > >::_data;

        void operator()()
        {
            // DEBUG
            //static int i = 0;
            //std::cout << Node::comm().rank() << "-" << i++ << " random: " << eo::rng.rand() << std::endl;

            // std::cout << "POP(" << _data->pop.size() << ") : " << _data->pop << std::endl;

            _data->resetAlgo();
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
        typedef eoUF< eoPop<EOT>&, void > ReinitJob;
        typedef eoUF< int, std::vector<int> > GetSeeds;

        MultiStartStore(
                eoAlgo<EOT> & algo,
                int masterRank,
                // eoInit<EOT>* init = 0
                ReinitJob & reinitJob,
                ResetAlgo & resetAlgo,
                GetSeeds & getSeeds
                )
            : _data( Node::comm(), algo, masterRank, resetAlgo ),
            _masterRank( masterRank ),
            _getSeeds( getSeeds ),
            _reinitJob( reinitJob )
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
            int nbWorkers = workers.size();

            _reinitJob( _data.pop );
            _data.runs = runs;

            std::vector< int > seeds = _getSeeds( nbWorkers );
            if( Node::comm().rank() == _masterRank )
            {
                if( seeds.size() < nbWorkers )
                {
                    // TODO
                    // get multiples of the current seed?
                    // generate seeds?
                    for( int i = 1; seeds.size() < nbWorkers ; ++i )
                    {
                        seeds.push_back( i );
                    }
                }

                for( int i = 0 ; i < nbWorkers ; ++i )
                {
                    int wrkRank = workers[i];
                    Node::comm().send( wrkRank, 1, seeds[ i ] );
                }
            } else
            {
                int seed;
                Node::comm().recv( _masterRank, 1, seed );
                std::cout << Node::comm().rank() << "- Seed: " << seed << std::endl;
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
        ReinitJob & _reinitJob;
        int _masterRank;
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

    EOT& best_individual()
    {
        return this->store.data()->bestIndividual;
    }

    typename EOT::Fitness best_fitness()
    {
        return this->store.data()->bestFitness;
    }
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
    }
};

template<class EOT>
struct RecopyPopEA : public MultiStartStore<EOT>::ReinitJob
{
    RecopyPopEA( const eoPop<EOT>& pop, eoEvalFunc<EOT>& eval ) : _originalPop( pop ), _eval( eval )
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
    }

    private:
    const eoPop<EOT>& _originalPop;
    eoEvalFunc<EOT>& _eval;
};

template<class EOT>
struct ResetGenContinueEA: public MultiStartStore<EOT>::ResetAlgo
{
    ResetGenContinueEA( eoGenContinue<EOT> & continuator ) :
        _continuator( continuator ),
        _initial( continuator.totalGenerations() )
    {
        // empty
    }

    void operator()()
    {
        _continuator.totalGenerations( _initial );
    }

    private:
    unsigned int _initial;
    eoGenContinue<EOT> & _continuator;
};

template< class EOT >
struct eoInitAndEval : public eoInit<EOT>
{
    eoInitAndEval( eoInit<EOT>& init, eoEvalFunc<EOT>& eval ) : _init( init ), _eval( eval )
    {
        // empty
    }

    void operator()( EOT & indi )
    {
        _init( indi );
        _eval( indi );
    }

    private:
    eoInit<EOT>& _init;
    eoEvalFunc<EOT>& _eval;
};

int main(int argc, char **argv)
{
    Node::init( argc, argv );

    // PARAMETRES
    // all parameters are hard-coded!
    const unsigned int SEED = 133742; // seed for random number generator
    const unsigned int VEC_SIZE = 8; // Number of object variables in genotypes
    const unsigned int POP_SIZE = 20; // Size of population
    const unsigned int T_SIZE = 3; // size for tournament selection
    const unsigned int MAX_GEN = 20; // Maximum number of generation before STOP
    const float CROSS_RATE = 0.8; // Crossover rate
    const double EPSILON = 0.01;  // range for real uniform mutation
    const float MUT_RATE = 0.5;   // mutation rate

    // GENERAL
    //////////////////////////
    //  Random seed
    //////////////////////////
    //reproducible random seed: if you don't change SEED above,
    // you'll aways get the same result, NOT a random run
    rng.reseed(SEED);

    // EVAL
    /////////////////////////////
    // Fitness function
    ////////////////////////////
    // Evaluation: from a plain C++ fn to an EvalFunc Object
    eoEvalFuncPtr<Indi> eval( real_value );

    // INIT
    ////////////////////////////////
    // Initilisation of population
    ////////////////////////////////

    // declare the population
    eoPop<Indi> pop;
    // fill it!
    /*
    for (unsigned int igeno=0; igeno<POP_SIZE; igeno++)
    {
        Indi v;          // void individual, to be filled
        for (unsigned ivar=0; ivar<VEC_SIZE; ivar++)
        {
            double r = 2*rng.uniform() - 1; // new value, random in [-1,1)
            v.push_back(r);       // append that random value to v
        }
        eval(v);                  // evaluate it
        pop.push_back(v);         // and put it in the population
    }
    */
    eoUniformGenerator< double > generator;
    eoInitFixedLength< Indi > init( VEC_SIZE, generator );
    // eoInitAndEval< Indi > init( real_init, eval, continuator );
    pop = eoPop<Indi>( POP_SIZE, init );

    // ENGINE
    /////////////////////////////////////
    // selection and replacement
    ////////////////////////////////////
    // SELECT
    // The robust tournament selection
    eoDetTournamentSelect<Indi> select(T_SIZE);       // T_SIZE in [2,POP_SIZE]

    // REPLACE
    // eoSGA uses generational replacement by default
    // so no replacement procedure has to be given

    // OPERATORS
    //////////////////////////////////////
    // The variation operators
    //////////////////////////////////////
    // CROSSOVER
    // offspring(i) is a linear combination of parent(i)
    eoSegmentCrossover<Indi> xover;
    // MUTATION
    // offspring(i) uniformly chosen in [parent(i)-epsilon, parent(i)+epsilon]
    eoUniformMutation<Indi>  mutation(EPSILON);

    // STOP
    // CHECKPOINT
    //////////////////////////////////////
    // termination condition
    /////////////////////////////////////
    // stop after MAX_GEN generations
    eoGenContinue<Indi> continuator(MAX_GEN); /** TODO FIXME FIXME BUG HERE!
                                                Continuator thinks it's done! */

    // GENERATION
    /////////////////////////////////////////
    // the algorithm
    ////////////////////////////////////////
    // standard Generational GA requires
    // selection, evaluation, crossover and mutation, stopping criterion

    eoSGA<Indi> gga(select, xover, CROSS_RATE, mutation, MUT_RATE,
            eval, continuator);

    DynamicAssignmentAlgorithm assignmentAlgo;
    MultiStartStore< Indi > store(
            gga,
            DEFAULT_MASTER,
            *new RecopyPopEA< Indi >( pop, eval ),
            *new ResetGenContinueEA< Indi >( continuator ),
            *new DummyGetSeeds< Indi >());

    MultiStart< Indi > msjob( assignmentAlgo, DEFAULT_MASTER, store, 5 );
    msjob.run();

    if( msjob.isMaster() )
    {
        std::cout << "Global best individual has fitness " << msjob.best_fitness() << std::endl;
    }

    MultiStart< Indi > msjob10( assignmentAlgo, DEFAULT_MASTER, store, 10 );
    msjob10.run();

    return 0;
}
