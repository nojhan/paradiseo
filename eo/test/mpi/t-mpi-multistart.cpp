# include <mpi/eoMultiStart.h>
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

    // GENERAL
    //////////////////////////
    //  Random seed
    //////////////////////////
    //reproducible random seed: if you don't change SEED above,
    // you'll aways get the same result, NOT a random run
    // rng.reseed(SEED);

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
    eoGenContinue<Indi> continuator(MAX_GEN);
    // eoSteadyFitContinue< Indi > continuator( 10, 50 );

    // GENERATION
    /////////////////////////////////////////
    // the algorithm
    ////////////////////////////////////////
    // standard Generational GA requires
    // selection, evaluation, crossover and mutation, stopping criterion

    eoSGA<Indi> gga(select, xover, CROSS_RATE, mutation, MUT_RATE,
            eval, continuator);

    DynamicAssignmentAlgorithm assignmentAlgo;
    ReuseSamePopEA< Indi > resetAlgo( continuator, pop, eval );
    GetRandomSeeds< Indi > getSeeds( SEED );

    MultiStartStore< Indi > store(
            gga,
            DEFAULT_MASTER,
            resetAlgo,
            getSeeds);

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
