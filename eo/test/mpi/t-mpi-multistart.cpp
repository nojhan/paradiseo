# include <mpi/eoMpi.h>
using namespace eo::mpi;

#include <eo>

/***************************** EASY PSO STUFF ********************************/
//-----------------------------------------------------------------------------
typedef eoMinimizingFitness ParticleFitness;
//-----------------------------------------------------------------------------
class SerializableParticle : public eoRealParticle< ParticleFitness >, public eoserial::Persistent
{
    public:

        SerializableParticle(unsigned size = 0, double positions = 0.0,double velocities = 0.0,double bestPositions = 0.0): eoRealParticle< ParticleFitness > (size, positions,velocities,bestPositions) {}

        void unpack( const eoserial::Object* obj )
        {
            this->clear();
            eoserial::unpackArray
                < std::vector<double>, eoserial::Array::UnpackAlgorithm >
                ( *obj, "vector", *this );

            this->bestPositions.clear();
            eoserial::unpackArray
                < std::vector<double>, eoserial::Array::UnpackAlgorithm >
                ( *obj, "best_positions", this->bestPositions );

            this->velocities.clear();
            eoserial::unpackArray
                < std::vector<double>, eoserial::Array::UnpackAlgorithm >
                ( *obj, "velocities", this->velocities );

            bool invalidFitness;
            eoserial::unpack( *obj, "invalid_fitness", invalidFitness );
            if( invalidFitness )
            {
                this->invalidate();
            } else
            {
                ParticleFitness f;
                eoserial::unpack( *obj, "fitness", f );
                this->fitness( f );
            }
        }

        eoserial::Object* pack( void ) const
        {
            eoserial::Object* obj = new eoserial::Object;
            obj->add( "vector", eoserial::makeArray< std::vector<double>, eoserial::MakeAlgorithm >( *this ) );
            obj->add( "best_positions", eoserial::makeArray< std::vector<double>, eoserial::MakeAlgorithm >( this->bestPositions ) );
            obj->add( "velocities", eoserial::makeArray< std::vector<double>, eoserial::MakeAlgorithm>( this->velocities ) );

            bool invalidFitness = this->invalid();
            obj->add( "invalid_fitness", eoserial::make( invalidFitness ) );
            if( !invalidFitness )
            {
                obj->add( "fitness", eoserial::make( this->fitness() ) );
            }

            return obj;
        }

};
typedef SerializableParticle Particle;
//-----------------------------------------------------------------------------

// the objective function
double real_value (const Particle & _particle)
{
    double sum = 0;
    for (unsigned i = 0; i < _particle.size ()-1; i++)
    sum += pow(_particle[i],2);
    return (sum);
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

template< class EOT, class FitT >
struct MultiStartData
{
    MultiStartData( mpi::communicator& _comm, eoAlgo<EOT>& _algo, int _masterRank, eoInit<EOT>* _init = 0 )
        :
        runs( 0 ), firstIndividual( true ), bestFitness(), pop(),
        comm( _comm ), algo( _algo ), masterRank( _masterRank ), init( _init )
    {
        // empty
    }

    // dynamic parameters
    int runs;
    bool firstIndividual;
    FitT bestFitness;
    EOT bestIndividual;
    eoPop< EOT > pop;

    // static parameters
    mpi::communicator& comm;
    eoAlgo<EOT>& algo;
    eoInit<EOT>* init;
    int masterRank;
};

template< class EOT, class FitT >
class SendTaskMultiStart : public SendTaskFunction< MultiStartData< EOT, FitT > >
{
    public:
        using SendTaskFunction< MultiStartData< EOT, FitT > >::_data;

        void operator()( int wrkRank )
        {
            --(_data->runs);
        }
};

template< class EOT, class FitT >
class HandleResponseMultiStart : public HandleResponseFunction< MultiStartData< EOT, FitT > >
{
    public:
        using HandleResponseFunction< MultiStartData< EOT, FitT > >::_data;

        void operator()( int wrkRank )
        {
            std::cout << "Response received." << std::endl;

            EOT individual;
            MultiStartData< EOT, FitT >& d = *_data;
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

template< class EOT, class FitT >
class ProcessTaskMultiStart : public ProcessTaskFunction< MultiStartData< EOT, FitT > >
{
    public:
        using ProcessTaskFunction< MultiStartData<EOT, FitT > >::_data;

        void operator()()
        {
            _data->algo( _data->pop );
            _data->comm.send( _data->masterRank, 1, _data->pop.best_element() );
        }
};

template< class EOT, class FitT >
class IsFinishedMultiStart : public IsFinishedFunction< MultiStartData< EOT, FitT > >
{
    public:
        using IsFinishedFunction< MultiStartData< EOT, FitT > >::_data;

        bool operator()()
        {
            return _data->runs <= 0;
        }
};

template< class EOT, class FitT >
class MultiStartStore : public JobStore< MultiStartData< EOT, FitT > >
{
    public:

        MultiStartStore( eoAlgo<EOT> & algo, int masterRank, const eoPop< EOT > & pop, eoInit<EOT>* init = 0 )
            : _data( Node::comm(), algo, masterRank, init ),
              _pop( pop ),
              _firstPopInit( true )
        {
            this->_iff = new IsFinishedMultiStart< EOT, FitT >;
            this->_iff->needDelete(true);
            this->_stf = new SendTaskMultiStart< EOT, FitT >;
            this->_stf->needDelete(true);
            this->_hrf = new HandleResponseMultiStart< EOT, FitT >;
            this->_hrf->needDelete(true);
            this->_ptf = new ProcessTaskMultiStart< EOT, FitT >;
            this->_ptf->needDelete(true);
        }

        void init( int runs )
        {
            _data.runs = runs;

            if( _data.init )
            {
                _data.pop = eoPop<EOT>( _pop.size(), *_data.init );
            } else if( _firstPopInit )
            {
                _data.pop = _pop;
            }
            _firstPopInit = false;

            _data.firstIndividual = true;
        }

        MultiStartData<EOT, FitT>* data()
        {
            return &_data;
        }

    private:
        MultiStartData< EOT, FitT > _data;
        const eoPop< EOT >& _pop;
        bool _firstPopInit;
};

template< class EOT, class FitT >
class MultiStart : public MultiJob< MultiStartData< EOT, FitT > >
{
    public:

        MultiStart( AssignmentAlgorithm & algo,
                int masterRank,
                MultiStartStore< EOT, FitT > & store,
                // dynamic parameters
                int runs,
                const std::vector<int>& seeds = std::vector<int>()  ) :
            MultiJob< MultiStartData< EOT, FitT > >( algo, masterRank, store )
    {
        store.init( runs );

        if( this->isMaster() )
        {
            int nbWorkers = algo.availableWorkers();
            std::vector<int> realSeeds = seeds;
            if( realSeeds.size() < nbWorkers )
            {
                // TODO
                // get multiples of the current seed?
                // generate seeds?
                for( int i = 1; realSeeds.size() < nbWorkers ; ++i )
                {
                    realSeeds.push_back( i );
                }
            }

            std::vector<int> idles = algo.idles();
            for( int i = 0 ; i < nbWorkers ; ++i )
            {
                int wrkRank = idles[i];
                Node::comm().send( wrkRank, 1, realSeeds[ i ] );
            }
        } else
        {
            int seed;
            Node::comm().recv( masterRank, 1, seed );
            std::cout << Node::comm().rank() << "- Seed: " << seed << std::endl;
            eo::rng.reseed( seed );
        }
    }

    EOT& best_individual()
    {
        return this->store.data()->bestIndividual;
    }

    FitT best_fitness()
    {
        return this->store.data()->bestFitness;
    }
};

int main(int argc, char **argv)
{
    Node::init( argc, argv );

    const unsigned int VEC_SIZE = 2;
    const unsigned int POP_SIZE = 20;
    const unsigned int NEIGHBORHOOD_SIZE= 5;
    unsigned i;

    eo::rng.reseed(1);

    // the population:
    eoPop<Particle> pop;

    // Evaluation
    eoEvalFuncPtr<Particle, double, const Particle& > eval(  real_value );

    // position init
    eoUniformGenerator < double >uGen (-3, 3);
    eoInitFixedLength < Particle > random (VEC_SIZE, uGen);

    // velocity init
    eoUniformGenerator < double >sGen (-2, 2);
    eoVelocityInitFixedLength < Particle > veloRandom (VEC_SIZE, sGen);

    // local best init
    eoFirstIsBestInit < Particle > localInit;

    // perform position initialization
    pop.append (POP_SIZE, random);

    // topology
    eoLinearTopology<Particle> topology(NEIGHBORHOOD_SIZE);

    // the full initializer
    eoInitializer <Particle> init(eval,veloRandom,localInit,topology,pop);
    init();

    // bounds
    eoRealVectorBounds bnds(VEC_SIZE,-1.5,1.5);

    // velocity
    eoStandardVelocity <Particle> velocity (topology,1,1.6,2,bnds);

    // flight
    eoStandardFlight <Particle> flight;

    // Terminators
    eoGenContinue <Particle> genCont1 (50);
    eoGenContinue <Particle> genCont2 (50);

    // PS flight
    eoEasyPSO<Particle> pso1(genCont1, eval, velocity, flight);

    // eoEasyPSO<Particle> pso2(init,genCont2, eval, velocity, flight);

    DynamicAssignmentAlgorithm assignmentAlgo;
    MultiStartStore< Particle, ParticleFitness > store( pso1, DEFAULT_MASTER, pop );

    MultiStart< Particle, ParticleFitness > msjob( assignmentAlgo, DEFAULT_MASTER, store, 5 );
    msjob.run();

    if( msjob.isMaster() )
    {
        eo::mpi::EmptyJob tjob( assignmentAlgo, DEFAULT_MASTER );
        std::cout << "Global best individual has fitness " << msjob.best_fitness() << std::endl;
    }

    // flight
    /*
    try
    {
        pso1(pop);
        std::cout << "FINAL POPULATION AFTER PSO n°1:" << std::endl;
        for (i = 0; i < pop.size(); ++i)
            std::cout << "\t" <<  pop[i] << " " << pop[i].fitness() << std::endl;

        pso2(pop);
        std::cout << "FINAL POPULATION AFTER PSO n°2:" << std::endl;
        for (i = 0; i < pop.size(); ++i)
            std::cout << "\t" <<  pop[i] << " " << pop[i].fitness() << std::endl;
    }
    catch (std::exception& e)
    {
        std::cout << "exception: " << e.what() << std::endl;;
        exit(EXIT_FAILURE);
    }
    */

    return 0;

}
