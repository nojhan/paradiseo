//-----------------------------------------------------------------------------
// FirstBitGA.cpp
//-----------------------------------------------------------------------------
//*
// An instance of a VERY simple Bitstring Genetic Algorithm
//
//-----------------------------------------------------------------------------

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdexcept>
#include <iostream>

#include <eo>
#include <ga.h>

// Use functions from namespace std
using namespace std;

// REPRESENTATION
//-----------------------------------------------------------------------------
// define your individuals
typedef eoBit<double> Indi;     // A bitstring with fitness double

// EVAL
//-----------------------------------------------------------------------------
// a simple fitness function that computes the number of ones of a bitstring
//  @param _indi A biststring individual

double binary_value(const Indi & _indi)
{
  double sum = 0;
  for (unsigned i = 0; i < _indi.size(); i++)
    sum += _indi[i];
  return sum;
}
// GENERAL
//-----------------------------------------------------------------------------
void main_function(int /*argc*/, char **/*argv*/)
{
// PARAMETRES
  // all parameters are hard-coded!
  const unsigned int SEED = 42;      // seed for random number generator
  const unsigned int T_SIZE = 3;     // size for tournament selection
  const unsigned int VEC_SIZE = 16;   // Number of bits in genotypes
  const unsigned int POP_SIZE = 100;  // Size of population
  const unsigned int MAX_GEN = 400;  // Maximum number of generation before STOP
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
  eoEvalFuncPtr<Indi> eval(  binary_value );

// INIT
  ////////////////////////////////
  // Initilisation of population
  ////////////////////////////////

  // declare the population
  eoPop<Indi> pop;
  // fill it!
  for (unsigned int igeno=0; igeno<POP_SIZE; igeno++)
    {
      Indi v;           // void individual, to be filled
      for (unsigned ivar=0; ivar<VEC_SIZE; ivar++)
	{
	  bool r = rng.flip(); // new value, random in {0,1}
	  v.push_back(r);      // append that random value to v
	}
      eval(v);                 // evaluate it
      pop.push_back(v);        // and put it in the population
    }

// OUTPUT
  // sort pop before printing it!
  pop.sort();
  // Print (sorted) intial population (raw printout)
  cout << "Initial Population" << endl;
  cout << pop;
  // shuffle  - this is a test
  pop.shuffle();
  // Print (sorted) intial population (raw printout)
  cout << "Shuffled Population" << endl;
  cout << pop;

// ENGINE
  /////////////////////////////////////
  // selection and replacement
  ////////////////////////////////////
// SELECT
  // The robust tournament selection
  eoDetTournamentSelect<Indi> select(T_SIZE);  // T_SIZE in [2,POP_SIZE]

// REPLACE
  // The simple GA evolution engine uses generational replacement
  // so no replacement procedure is needed

// OPERATORS
  //////////////////////////////////////
  // The variation operators
  //////////////////////////////////////
// CROSSOVER
  // 1-point crossover for bitstring
  eo1PtBitXover<Indi> xover;
// MUTATION
  // standard bit-flip mutation for bitstring
  eoBitMutation<Indi>  mutation(P_MUT_PER_BIT);

// STOP
// CHECKPOINT
  //////////////////////////////////////
  // termination condition
  /////////////////////////////////////
  // stop after MAX_GEN generations
  eoGenContinue<Indi> continuator(MAX_GEN);

// GENERATION
  /////////////////////////////////////////
  // the algorithm
  ////////////////////////////////////////
  // standard Generational GA requires as parameters
  // selection, evaluation, crossover and mutation, stopping criterion


  eoSGA<Indi> gga(select, xover, CROSS_RATE, mutation, MUT_RATE,
		  eval, continuator);

  // Apply algo to pop - that's it!
  gga(pop);

// OUTPUT
  // Print (sorted) intial population
  pop.sort();
  cout << "FINAL Population\n" << pop << endl;
// GENERAL
}
 // A main that catches the exceptions

int main(int argc, char **argv)
{

    try
    {
	main_function(argc, argv);
    }
    catch(exception& e)
    {
	cout << "Exception: " << e.what() << '\n';
    }

    return 1;
}
