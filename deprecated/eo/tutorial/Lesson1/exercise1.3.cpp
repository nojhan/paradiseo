#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

//-----------------------------------------------------------------------------
// FirstBitGA.cpp
//-----------------------------------------------------------------------------
//*
// An instance of a VERY simple Bitstring Genetic Algorithm
//
//-----------------------------------------------------------------------------
// standard includes
#include <iostream>
#include <stdexcept>

// the general include for eo
#include <eo>

//-----------------------------------------------------------------------------
// Include the corresponding file
#include <ga.h>		 // bitstring representation & operators
// define your individuals
typedef eoBit<double> Indi;	// A bitstring with fitness double

using namespace std;

//-----------------------------------------------------------------------------
/** a simple fitness function that computes the number of ones of a bitstring
    @param _indi A biststring individual
*/

double binary_value(const Indi & _indi)
{
  double sum = 0;
  for (unsigned i = 0; i < _indi.size(); i++)
    sum += _indi[i];
  return sum;
}

//-----------------------------------------------------------------------------

void main_function(int argc, char **argv)
{
  const unsigned int SEED = 42;	// seed for random number generator
  const unsigned int VEC_SIZE = 8; // Number of bits in genotypes
  const unsigned int POP_SIZE = 20; // Size of population
  const unsigned int MAX_GEN = 500; // Maximum number of generation before STOP
  const float CROSS_RATE = 0.8;	// Crossover rate
  const double P_MUT_PER_BIT = 0.01;	// probability of bit-flip mutation
  const float MUT_RATE = 1.0;	// mutation rate

  //////////////////////////
  //  Random seed
  //////////////////////////
  //reproducible random seed: if you don't change SEED above,
  // you'll aways get the same result, NOT a random run
  rng.reseed(SEED);

  /////////////////////////////
  // Fitness function
  ////////////////////////////
  // Evaluation: from a plain C++ fn to an EvalFunc Object
  eoEvalFuncPtr<Indi> eval(  binary_value );

  ////////////////////////////////
  // Initilisation of population
  ////////////////////////////////

  // declare the population
  eoPop<Indi> pop;
  // fill it!
  for (unsigned int igeno=0; igeno<POP_SIZE; igeno++)
    {
      Indi v;		// void individual, to be filled
      for (unsigned ivar=0; ivar<VEC_SIZE; ivar++)
	{
	  bool r = rng.flip(); // new value, random in {0,1}
	  v.push_back(r);	// append that random value to v
	}
      eval(v);			// evaluate it
      pop.push_back(v);		// and put it in the population
    }

  // sort pop before printing it!
  pop.sort();
  // Print (sorted) intial population (raw printout)
  cout << "Initial Population" << endl;
  cout << pop;

  /////////////////////////////////////
  // selection and replacement
  ////////////////////////////////////

  // solution solution solution: uncomment one of the following,
  //                             comment out the eoDetTournament lines

  // The well-known roulette
  // eoProportionalSelect<Indi> select;

  // could also use stochastic binary tournament selection
  //
  //  const double RATE = 0.75;
  //  eoStochTournamentSelect<Indi> select(RATE);     // RATE in ]0.5,1]
  // The robust tournament selection
  const unsigned int T_SIZE = 3; // size for tournament selection
  eoDetTournamentSelect<Indi> select(T_SIZE);       // T_SIZE in [2,POP_SIZE]

  // and of course the random selection
  // eoRandomSelect<Indi> select;

  // The simple GA evolution engine uses generational replacement
  // so no replacement procedure is needed

  //////////////////////////////////////
  // termination condition
  /////////////////////////////////////
  // stop after MAX_GEN generations
  eoGenContinue<Indi> continuator(MAX_GEN);


  //////////////////////////////////////
  // The variation operators
  //////////////////////////////////////
  // standard bit-flip mutation for bitstring
  eoBitMutation<Indi>  mutation(P_MUT_PER_BIT);
  // 1-point mutation for bitstring
  eo1PtBitXover<Indi> xover;

  /////////////////////////////////////////
  // the algorithm
  ////////////////////////////////////////
  // standard Generational GA requires as parameters
  // selection, evaluation, crossover and mutation, stopping criterion


  eoSGA<Indi> gga(select, xover, CROSS_RATE, mutation, MUT_RATE,
  		   eval, continuator);

  // Apply algo to pop - that's it!
  gga(pop);

  // Print (sorted) intial population
  pop.sort();
  cout << "FINAL Population\n" << pop << endl;
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
