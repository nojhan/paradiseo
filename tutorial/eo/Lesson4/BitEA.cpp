#include <iostream>

#include <paradiseo/eo/ga/make_ga.h>
#include <paradiseo/eo/apply.h>

// EVAL
#include "binary_value.h"

// GENERAL
using namespace std;

int main(int argc, char* argv[])
{

  try
  {
// REPRESENTATION
//-----------------------------------------------------------------------------
// define your genotype and fitness types
  typedef eoBit<double> EOT;

// PARAMETRES
  eoParser parser(argc, argv);  // for user-parameter reading

// GENERAL
  eoState state;    // keeps all things allocated

  ///// FIRST, problem or representation dependent stuff
  //////////////////////////////////////////////////////

// EVAL
  // The evaluation fn - encapsulated into an eval counter for output
  eoEvalFuncPtr<EOT, double> mainEval( binary_value<EOT> );
  eoEvalFuncCounter<EOT> eval(mainEval);

// REPRESENTATION
  // the genotype - through a genotype initializer
  eoInit<EOT>& init = make_genotype(parser, state, EOT());

  // if you want to do sharing, you'll need a distance.
  // here Hamming distance
  eoHammingDistance<EOT> dist;

// OPERATORS
  // Build the variation operator (any seq/prop construct)
  eoGenOp<EOT>& op = make_op(parser, state, init);

// GENERAL
  //// Now the representation-independent things
  //////////////////////////////////////////////

  // initialize the population - and evaluate
  // yes, this is representation indepedent once you have an eoInit
  eoPop<EOT>& pop   = make_pop(parser, state, init);

// STOP
  // stopping criteria
  eoContinue<EOT> & term = make_continue(parser, state, eval);
  // output
  eoCheckPoint<EOT> & checkpoint = make_checkpoint(parser, state, eval, term);
// GENERATION
  // algorithm (need the operator!)
  eoAlgo<EOT>& ga = make_algo_scalar(parser, state, eval, checkpoint, op, &dist);

  ///// End of construction of the algorith
  /////////////////////////////////////////
// PARAMETRES
  // to be called AFTER all parameters have been read!!!
  make_help(parser);

  //// GO
  ///////
// EVAL
  // evaluate intial population AFTER help and status in case it takes time
  apply<EOT>(eval, pop);
// STOP
  // print it out (sort witout modifying)
  cout << "Initial Population\n";
  pop.sortedPrintOn(cout);
  cout << endl;

// GENERATION
  run_ea(ga, pop); // run the ga
// STOP
  // print it out (sort witout modifying)
  cout << "Final Population\n";
  pop.sortedPrintOn(cout);
  cout << endl;
// GENERAL
  }
  catch(exception& e)
  {
    cout << e.what() << endl;
  }
}
