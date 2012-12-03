#include <smp>
#include <eo>
#include <ga.h>

#include "smpTestClass.h"

using namespace paradiseo::smp;
using namespace std;

typedef eoBit<double> Indi2;     // A bitstring with fitness double

// Conversion functions
Indi2 fromBase(Indi& i, unsigned size)
{
    // Dummy conversion. We just create a new Indi2
    Indi2 v;
    for (unsigned ivar=0; ivar<size; ivar++)
	{
	      bool r = rng.flip(); // new value, random in {0,1}
	      v.push_back(r);      // append that random value to v
	}
    std::cout << "Convert from base : " << v << std::endl;
    return v;
}

Indi toBase(Indi2& i)
{
    // Dummy conversion. We just create a new Indi
    Indi v;
    std::cout << "Convert to base : " << v << std::endl;
    return v;
}

// EVAL
//-----------------------------------------------------------------------------
// a simple fitness function that computes the number of ones of a bitstring
//  @param _Indi2 A biststring Indi2vidual

double binary_value(const Indi2 & _Indi2)
{
  double sum = 0;
  for (unsigned i = 0; i < _Indi2.size(); i++)
    sum += _Indi2[i];
  return sum;
}
// GENERAL
//-----------------------------------------------------------------------------

int main(void)
{
// PARAMETRES
  // all parameters are hard-coded!
  const unsigned int SEED = 42;      // seed for random number generator
  const unsigned int T_SIZE = 3;     // size for tournament selection
  const unsigned int VEC_SIZE = 16;   // Number of bits in genotypes
  const unsigned int POP_SIZE = 10;  // Size of population
  const unsigned int MAX_GEN = 10;  // Maximum number of generation before STOP
  const float CROSS_RATE = 0.8;      // Crossover rate
  const double P_MUT_PER_BIT = 0.01; // probability of bit-flip mutation
  const float MUT_RATE = 1.0;        // mutation rate

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
  eoEvalFuncPtr<Indi2> eval(  binary_value );

// INIT
  ////////////////////////////////
  // Initilisation of population
  ////////////////////////////////

  // declare the population
  eoPop<Indi2> pop;
  // fill it!
  for (unsigned int igeno=0; igeno<POP_SIZE; igeno++)
    {
      Indi2 v;           // void Indi2vidual, to be filled
      for (unsigned ivar=0; ivar<VEC_SIZE; ivar++)
	{
	  bool r = rng.flip(); // new value, random in {0,1}
	  v.push_back(r);      // append that random value to v
	}
      eval(v);                 // evaluate it
      pop.push_back(v);        // and put it in the population
    }

// ENGINE
  /////////////////////////////////////
  // selection and replacement
  ////////////////////////////////////
// SELECT
  // The robust tournament selection
  eoDetTournamentSelect<Indi2> select(T_SIZE);  // T_SIZE in [2,POP_SIZE]

// REPLACE
  // The simple GA evolution engine uses generational replacement
  // so no replacement procedure is needed

// OPERATORS
  //////////////////////////////////////
  // The variation operators
  //////////////////////////////////////
// CROSSOVER
  // 1-point crossover for bitstring
  eo1PtBitXover<Indi2> xover;
// MUTATION
  // standard bit-flip mutation for bitstring
  eoBitMutation<Indi2>  mutation(P_MUT_PER_BIT);

// STOP
// CHECKPOINT
  //////////////////////////////////////
  // termination condition
  /////////////////////////////////////
  // stop after MAX_GEN generations
  eoGenContinue<Indi2> continuator(MAX_GEN);

// GENERATION
  /////////////////////////////////////////
  // the algorithm
  ////////////////////////////////////////
  // standard Generational GA requires as parameters
  // selection, evaluation, crossover and mutation, stopping criterion

// // Emigration policy
        // // // Element 1 
        eoPeriodicContinue<Indi2> criteria(1);
        eoDetTournamentSelect<Indi2> selectOne(2);
        eoSelectNumber<Indi2> who(selectOne, 1);
        
        MigPolicy<Indi2> migPolicy;
        migPolicy.push_back(PolicyElement<Indi2>(who, criteria));
        
        // // Integration policy
        eoPlusReplacement<Indi2> intPolicy;
        
        // We bind conversion functions
        auto frombase = std::bind(fromBase, std::placeholders::_1, VEC_SIZE);
        auto tobase = std::bind(toBase, std::placeholders::_1);

  Island<eoSGA,Indi2, Indi> gga(frombase, tobase, pop, intPolicy, migPolicy, select, xover, CROSS_RATE, mutation, MUT_RATE,
		  eval, continuator);
//////////////////////////////////////////////////////////////////
        typedef struct {
        unsigned popSize = 10;
        unsigned tSize = 2;
        double pCross = 0.8;
        double pMut = 0.7;
        unsigned maxGen = 10;
    } Param; 

    Param param;

    loadInstances("t-data.dat", n, bkv, a, b);
      
    // Evaluation function
    IndiEvalFunc plainEval;
    
    // Init a solution
    IndiInit chromInit;
    
    // Define selection
    eoDetTournamentSelect<Indi> selectOne2(param.tSize);
    eoSelectPerc<Indi> select2(selectOne2);// by default rate==1

    // Define operators for crossover and mutation
    IndiXover Xover;                 // CROSSOVER
    IndiSwapMutation  mutationSwap;  // MUTATION
      
    // Encapsule in a tranform operator
    eoSGATransform<Indi> transform(Xover, param.pCross, mutationSwap, param.pMut);
    
    // Define replace operator
    eoPlusReplacement<Indi> replace;

    eoGenContinue<Indi> genCont(param.maxGen); // generation continuation
    
    // Define population
    eoPop<Indi> pop2(param.popSize, chromInit);

    // Island 1
    // // Emigration policy
    // // // Element 1 
    eoPeriodicContinue<Indi> criteria2(1);
    eoDetTournamentSelect<Indi> selectOne3(5);
    eoSelectNumber<Indi> who2(selectOne3, 2);
        
    MigPolicy<Indi> migPolicy2;
    migPolicy2.push_back(PolicyElement<Indi>(who2, criteria2));
        
    // // Integration policy
    eoPlusReplacement<Indi> intPolicy2;

    Island<eoEasyEA,Indi> test(pop2, intPolicy2, migPolicy2, genCont, plainEval, select2, transform, replace);
    
    // Topology
        Topology<Complete> topo;
        
        IslandModel<Indi> model(topo);
        model.add(test);
        model.add(gga);
        
        model();
        
        cout << test.getPop() << endl;
        cout << gga.getPop() << endl;


    return 0;
}
