#include <iostream>

#include <es/make_real.h>
#include "real_value.h"
#include <apply.h>

using namespace std;

int main(int argc, char* argv[])
{

  try
  {
  typedef eoReal<eoMinimizingFitness> EOT;

  eoParser parser(argc, argv);  // for user-parameter reading

  eoState state;    // keeps all things allocated

  ///// FIRST, problem or representation dependent stuff
  //////////////////////////////////////////////////////

  // The evaluation fn - encapsulated into an eval counter for output
  eoEvalFuncPtr<EOT, double, const std::vector<double>&>
	       mainEval( real_value );
  eoEvalFuncCounter<EOT> eval(mainEval);

  // the genotype - through a genotype initializer
  eoRealInitBounded<EOT>& init = make_genotype(parser, state, EOT());

  // Build the variation operator (any seq/prop construct)
  eoGenOp<EOT>& op = make_op(parser, state, init);

  //// Now the representation-independent things
  //////////////////////////////////////////////

  // initialize the population - and evaluate
  // yes, this is representation indepedent once you have an eoInit
  eoPop<EOT>& pop   = make_pop(parser, state, init);

  // stopping criteria
  eoContinue<EOT> & term = make_continue(parser, state, eval);
  // output
  eoCheckPoint<EOT> & checkpoint = make_checkpoint(parser, state, eval, term);
  // algorithm (need the operator!)
  eoAlgo<EOT>& ea = make_algo_scalar(parser, state, eval, checkpoint, op);

  ///// End of construction of the algorith
  /////////////////////////////////////////
  // to be called AFTER all parameters have been read!!!
  make_help(parser);

  //// GO
  ///////
  // evaluate intial population AFTER help and status in case it takes time
  apply<EOT>(eval, pop);
  // print it out
  std::cout << "Initial Population\n";
  pop.sortedPrintOn(std::cout);
  std::cout << std::endl;

  run_ea(ea, pop); // run the ea

  std::cout << "Final Population\n";
  pop.sortedPrintOn(std::cout);
  std::cout << std::endl;
  }
  catch(std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
}
