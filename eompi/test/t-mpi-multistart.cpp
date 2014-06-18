# include <eoMultiStart.h>
using namespace eo::mpi;

#include <stdexcept>
#include <iostream>
#include <sstream>

#include <eo>
#include <es.h>

/*
 * This file is based on the tutorial lesson 1. We'll consider that you know all the EO
 * related parts of the algorithm and we'll focus our attention on parallelization.
 *
 * This file shows an example of multistart applied to a eoSGA (simple genetic
 * algorithm). As individuals need to be serialized, we implement a class inheriting
 * from eoReal (which is the base individual), so as to manipulate individuals as they
 * were eoReal AND serialize them.
 *
 * The main function shows how to launch a multistart job, with default functors. If you
 * don't know which functors to use, these ones should fit the most of your purposes.
 */

using namespace std;

/*
 * eoReal is a vector of double: we just have to serializes the value and the fitness.
 */
class SerializableEOReal: public eoReal<double>, public eoserial::Persistent
{
public:

    SerializableEOReal(unsigned size = 0, double value = 0.0) :
            eoReal<double>(size, value)
    {
        // empty
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
int main(int argc, char **argv)
{
    Node::init( argc, argv );

    // PARAMETRES
    // all parameters are hard-coded!
    const unsigned int SEED = 133742; // seed for random number generator
    const unsigned int VEC_SIZE = 8; // Number of object variables in genotypes
    const unsigned int POP_SIZE = 100; // Size of population
    const unsigned int T_SIZE = 3; // size for tournament selection
    const unsigned int MAX_GEN = 100; // Maximum number of generation before STOP
    const float CROSS_RATE = 0.8; // Crossover rate
    const double EPSILON = 0.01;  // range for real uniform mutation
    const float MUT_RATE = 0.5;   // mutation rate

    eoEvalFuncPtr<Indi> eval( real_value );
    eoPop<Indi> pop;
    eoUniformGenerator< double > generator;
    eoInitFixedLength< Indi > init( VEC_SIZE, generator );
    pop = eoPop<Indi>( POP_SIZE, init );

    eoDetTournamentSelect<Indi> select(T_SIZE);
    eoSegmentCrossover<Indi> xover;
    eoUniformMutation<Indi>  mutation(EPSILON);

    eoGenContinue<Indi> continuator(MAX_GEN);
    /* Does work too with a steady fit continuator. */
    // eoSteadyFitContinue< Indi > continuator( 10, 50 );

    eoSGA<Indi> gga(select, xover, CROSS_RATE, mutation, MUT_RATE,
            eval, continuator);

    /* How to assign tasks, which are starts? */
    DynamicAssignmentAlgorithm assignmentAlgo;
    /* Before a worker starts its algorithm, how does it reinits the population?
     * There are a few default usable functors, defined in eoMultiStart.h.
     *
     * This one (ReuseSamePopEA) doesn't modify the population after a start, so
     * the same population is reevaluated on each multistart: the solution tend
     * to get better and better.
     */
    ReuseSamePopEA< Indi > resetAlgo( continuator, pop, eval );
    /**
     * How to send seeds to the workers, at the beginning of the parallel job?
     * This functors indicates that seeds should be random values.
     */
    GetRandomSeeds< Indi > getSeeds( SEED );

    // Builds the store
    MultiStartStore< Indi > store(
            gga,
            DEFAULT_MASTER,
            resetAlgo,
            getSeeds);

    // Creates the multistart job and runs it.
    // The last argument indicates that we want to launch 5 runs.
    MultiStart< Indi > msjob( assignmentAlgo, DEFAULT_MASTER, store, 5 );
    msjob.run();

    if( msjob.isMaster() )
    {
        msjob.best_individuals().sort();
        std::cout << "Global best individual has fitness " << msjob.best_individuals().best_element().fitness() << std::endl;
    }

    MultiStart< Indi > msjob10( assignmentAlgo, DEFAULT_MASTER, store, 10 );
    msjob10.run();

    if( msjob10.isMaster() )
    {
        msjob10.best_individuals().sort();
        std::cout << "Global best individual has fitness " << msjob10.best_individuals().best_element().fitness() << std::endl;
    }
    return 0;
}
