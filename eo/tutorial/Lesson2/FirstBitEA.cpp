//-----------------------------------------------------------------------------
// FirstBitEA.cpp
//-----------------------------------------------------------------------------
//*
// Still an instance of a VERY simple Bitstring Genetic Algorithm 
// (see FirstBitGA.cpp) but now with  Breeder - and Combined Ops
//
//-----------------------------------------------------------------------------
// standard includes
#include <stdexcept>  // runtime_error 
#include <iostream>   // cout
#include <strstream>  // ostrstream, istrstream

// the general include for eo
#include <eo>

// REPRESENTATION
//-----------------------------------------------------------------------------
// define your individuals
typedef eoBin<double> Indi;	// A bitstring with fitness double

// EVALFUNC
//-----------------------------------------------------------------------------
// a simple fitness function that computes the number of ones of a bitstring
// Now in a separate file, and declared as binary_value(const vector<bool> &)

#include "binary_value.h"

// GENERAL
//-----------------------------------------------------------------------------

void main_function(int argc, char **argv)
{
// PARAMETRES
  const unsigned int SEED = 42;	// seed for random number generator
  const unsigned int T_SIZE = 3; // size for tournament selection
  const unsigned int VEC_SIZE = 8; // Number of bits in genotypes
  const unsigned int POP_SIZE = 20; // Size of population

  const unsigned int MAX_GEN = 500; // Maximum number of generation before STOP
  const unsigned int MIN_GEN = 10;  // Minimum number of generation before ...
  const unsigned int STEADY_GEN = 50; // stop after STEADY_GEN gen. without improvelent

  const double P_CROSS = 0.8;	// Crossover probability
  const double P_MUT = 1.0;	// mutation probability

  const double P_MUT_PER_BIT = 0.01;	// internal probability for bit-flip mutation
  // some parameters for chosing among different operators
  const double onePointRate = 0.5;     // rate for 1-pt Xover
  const double twoPointsRate = 0.5;     // rate for 2-pt Xover
  const double URate = 0.5;            // rate for Uniform Xover
  const double bitFlipRate = 0.5;      // rate for bit-flip mutation
  const double oneBitRate = 0.5;       // rate for one-bit mutation

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
  // you need to give the full description of the function
  eoEvalFuncPtr<Indi, double, const vector<bool>& > eval(  binary_value );

// INIT
  ////////////////////////////////
  // Initilisation of population
  ////////////////////////////////

  // based on boolean_generator class (see utils/rnd_generator.h)
  eoInitFixedLength<Indi, boolean_generator> 
    random(VEC_SIZE, boolean_generator());
  // Initialization of the population
  eoPop<Indi> pop(POP_SIZE, random);

  // and evaluate it in one loop
  apply<Indi>(eval, pop);	// STL syntax

// OUTPUT
  // sort pop before printing it!
  pop.sort();
  // Print (sorted) intial population (raw printout)
  cout << "Initial Population" << endl;
  cout << pop;

// ENGINE
  /////////////////////////////////////
  // selection and replacement
  ////////////////////////////////////
// SELECT
  // The robust tournament selection
  eoDetTournament<Indi> selectOne(T_SIZE);       // T_SIZE in [2,POP_SIZE]
  // is now encapsulated in a eoSelectPerc (entage)
  eoSelectPerc<Indi> select(selectOne);// by default rate==1

// REPLACE
  // And we now have the full slection/replacement - though with 
  // no replacement (== generational replacement) at the moment :-)
  eoNoReplacement<Indi> replace; 

// OPERATORS
  //////////////////////////////////////
  // The variation operators
  //////////////////////////////////////
// CROSSOVER
  // 1-point crossover for bitstring
  eoBinCrossover<Indi> xover1;
  // uniform crossover for bitstring
  eoBinUxOver<Indi> xoverU;
  // 2-pots xover
  eoBinNxOver<Indi> xover2(2);
  // Combine them with relative rates
  eoPropCombinedQuadOp<Indi> xover(xover1, onePointRate);
  xover.add(xoverU, URate);
  xover.add(xover2, twoPointsRate, true);

// MUTATION
  // standard bit-flip mutation for bitstring
  eoBinMutation<Indi>  mutationBitFlip(P_MUT_PER_BIT);
  // mutate exactly 1 bit per individual
  eoDetBitFlip<Indi> mutationOneBit; 
  // Combine them with relative rates
  eoPropCombinedMonOp<Indi> mutation(mutationBitFlip, bitFlipRate);
  mutation.add(mutationOneBit, oneBitRate, true);

  // The operators are  encapsulated into an eoTRansform object
  eoSGATransform<Indi> transform(xover, P_CROSS, mutation, P_MUT);

// STOP
// CHECKPOINT
  //////////////////////////////////////
  // termination conditions: use more than one
  /////////////////////////////////////
  // stop after MAX_GEN generations
  eoGenContinue<Indi> genCont(MAX_GEN);
  // do MIN_GEN gen., then stop after STEADY_GEN gen. without improvement
  eoSteadyFitContinue<Indi> steadyCont(MIN_GEN, STEADY_GEN);
  // stop when fitness reaches a target (here VEC_SIZE)
  eoFitContinue<Indi> fitCont(VEC_SIZE);
  // do stop when one of the above says so
  eoCombinedContinue<Indi> continuator(genCont);
  continuator.add(steadyCont);
  continuator.add(fitCont);
  
// GENERATION
  /////////////////////////////////////////
  // the algorithm
  ////////////////////////////////////////

  // Easy EA requires 
  // selection, transformation, eval, replacement, and stopping criterion
  eoEasyEA<Indi> gga(continuator, eval, select, transform, replace);

  // Apply algo to pop - that's it!
  cout << "\n        Here we go\n\n";
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
#ifdef _MSC_VER
  //  rng.reseed(42);
    int flag = _CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF);
     flag |= _CRTDBG_LEAK_CHECK_DF;
    _CrtSetDbgFlag(flag);
//   _CrtSetBreakAlloc(100);
#endif

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
