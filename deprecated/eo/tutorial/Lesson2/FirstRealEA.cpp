//-----------------------------------------------------------------------------
// FirstRealEA.cpp
//-----------------------------------------------------------------------------
//*
// Still an instance of a VERY simple Real-coded  Genetic Algorithm
// (see FirstBitGA.cpp) but now with  Breeder - and Combined Ops
//
//-----------------------------------------------------------------------------
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

// standard includes
#include <stdexcept>  // runtime_error
#include <iostream>   // cout

// the general include for eo
#include <eo>
#include <es.h>

// REPRESENTATION
//-----------------------------------------------------------------------------
// define your individuals
typedef eoReal<double> Indi;

// Use functions from namespace std
using namespace std;

// EVALFUNC
//-----------------------------------------------------------------------------
// a simple fitness function that computes the euclidian norm of a real vector
// Now in a separate file, and declared as binary_value(const vector<bool> &)

#include "real_value.h"

// GENERAL
//-----------------------------------------------------------------------------

void main_function(int argc, char **argv)
{
// PARAMETRES
  const unsigned int SEED = 42;	// seed for random number generator
  const unsigned int T_SIZE = 3; // size for tournament selection
  const unsigned int VEC_SIZE = 8; // Number of object variables in genotypes
  const unsigned int POP_SIZE = 20; // Size of population

  const unsigned int MAX_GEN = 500; // Maximum number of generation before STOP
  const unsigned int MIN_GEN = 10;  // Minimum number of generation before ...
  const unsigned int STEADY_GEN = 50; // stop after STEADY_GEN gen. without improvelent

  const float P_CROSS = 0.8;	// Crossover probability
  const float P_MUT = 0.5;	// mutation probability

  const double EPSILON = 0.01;	// range for real uniform mutation
  double SIGMA = 0.3;	    	// std dev. for normal mutation
  // some parameters for chosing among different operators
  const double hypercubeRate = 0.5;     // relative weight for hypercube Xover
  const double segmentRate = 0.5;  // relative weight for segment Xover
  const double uniformMutRate = 0.5;  // relative weight for uniform mutation
  const double detMutRate = 0.5;      // relative weight for det-uniform mutation
  const double normalMutRate = 0.5;   // relative weight for normal mutation

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
  eoEvalFuncPtr<Indi, double, const vector<double>& > eval(  real_value );

// INIT
  ////////////////////////////////
  // Initilisation of population
  ////////////////////////////////
  // based on a uniform generator
  eoUniformGenerator<double> uGen(-1.0, 1.0);
  eoInitFixedLength<Indi> random(VEC_SIZE, uGen);
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
  eoDetTournamentSelect<Indi> selectOne(T_SIZE);
  // is now encapsulated in a eoSelectPerc (entage)
  eoSelectPerc<Indi> select(selectOne);// by default rate==1

// REPLACE
  // And we now have the full slection/replacement - though with
  // no replacement (== generational replacement) at the moment :-)
  eoGenerationalReplacement<Indi> replace;

// OPERATORS
  //////////////////////////////////////
  // The variation operators
  //////////////////////////////////////
// CROSSOVER
  // uniform chooce on segment made by the parents
  eoSegmentCrossover<Indi> xoverS;
  // uniform choice in hypercube built by the parents
  eoHypercubeCrossover<Indi> xoverA;
  // Combine them with relative weights
  eoPropCombinedQuadOp<Indi> xover(xoverS, segmentRate);
  xover.add(xoverA, hypercubeRate, true);

// MUTATION
  // offspring(i) uniformly chosen in [parent(i)-epsilon, parent(i)+epsilon]
  eoUniformMutation<Indi>  mutationU(EPSILON);
  // k (=1) coordinates of parents are uniformly modified
  eoDetUniformMutation<Indi>  mutationD(EPSILON);
  // all coordinates of parents are normally modified (stDev SIGMA)
  eoNormalMutation<Indi>  mutationN(SIGMA);
  // Combine them with relative weights
  eoPropCombinedMonOp<Indi> mutation(mutationU, uniformMutRate);
  mutation.add(mutationD, detMutRate);
  mutation.add(mutationN, normalMutRate, true);

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
  eoFitContinue<Indi> fitCont(0);
  // do stop when one of the above says so
  eoCombinedContinue<Indi> continuator(genCont);
  continuator.add(steadyCont);
  continuator.add(fitCont);

  // The operators are  encapsulated into an eoTRansform object
  eoSGATransform<Indi> transform(xover, P_CROSS, mutation, P_MUT);

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
