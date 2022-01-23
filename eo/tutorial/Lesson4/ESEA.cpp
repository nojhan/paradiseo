// Program to test several EO-ES features

#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif

#include <algorithm>
#include <string>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <time.h>

using namespace std;

#include <eo>

// representation specific
#include <es/make_es.h>

#include "real_value.h"		// the sphere fitness

// Now the main
///////////////
typedef eoMinimizingFitness  FitT;

template <class EOT>
void runAlgorithm(EOT, eoParser& _parser, eoState& _state);

int main_function(int argc, char *argv[])
{
  // Create the command-line parser
  eoParser parser(argc, argv);  // for user-parameter reading

  eoState state;    // keeps all things allocated


  eoValueParam<bool>& simpleParam = parser.createParam(true, "Isotropic", "Isotropic self-adaptive mutation", 'i', "ES mutation");
  eoValueParam<bool>& stdevsParam = parser.createParam(false, "Stdev", "One self-adaptive stDev per variable", 's', "ES mutation");
  eoValueParam<bool>& corrParam = parser.createParam(false, "Correl", "Use correlated mutations", 'c', "ES mutation");

    // Run the appropriate algorithm
    if (simpleParam.value() == false)
    {
      cout << "Using eoReal" << endl;
      runAlgorithm(eoReal<FitT>(), parser, state);
    }
    else if (stdevsParam.value() == false)
    {
      cout << "Using eoEsSimple" << endl;
      runAlgorithm(eoEsSimple<FitT>(), parser, state);
    }
    else if (corrParam.value() == false)
    {
      cout << "Using eoEsStdev" << endl;
      runAlgorithm(eoEsStdev<FitT>(), parser, state);
    }
    else
    {
      cout << "Using eoEsFull" << endl;
      runAlgorithm(eoEsFull<FitT>(), parser, state);
    }

	return 0;
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

/** The templatized main (sort of)
 *  quite similar to the main of other genotypes
 *  (e.g. t-eoReal and t-eoGA in test dir)
 */
template <class EOT>
void runAlgorithm(EOT, eoParser& _parser, eoState& _state)
{
  // typedef typename EOT::Fitness FitT;

  ///// FIRST, problem or representation dependent stuff
  //////////////////////////////////////////////////////

  // The evaluation fn - encapsulated into an eval counter for output
  eoEvalFuncPtr<EOT, double, const std::vector<double>&>
	       mainEval( real_value );
  eoEvalFuncCounter<EOT> eval(mainEval);

  // the genotype - through a genotype initializer
  eoRealInitBounded<EOT>& init = make_genotype(_parser, _state, EOT());

  // Build the variation operator (any seq/prop construct)
  eoGenOp<EOT>& op = make_op(_parser, _state, init);

  //// Now the representation-independent things
  //////////////////////////////////////////////

  // initialize the population - and evaluate
  // yes, this is representation indepedent once you have an eoInit
  eoPop<EOT>& pop   = make_pop(_parser, _state, init);
  apply<EOT>(eval, pop);

  // stopping criteria
  eoContinue<EOT> & term = make_continue(_parser, _state, eval);
  // output
  eoCheckPoint<EOT> & checkpoint = make_checkpoint(_parser, _state, eval, term);
  // algorithm (need the operator!)
  eoAlgo<EOT>& ga = make_algo_scalar(_parser, _state, eval, checkpoint, op);

  ///// End of construction of the algorith
  /////////////////////////////////////////
  // to be called AFTER all parameters have been read!!!
  make_help(_parser);

  //// GO
  ///////
  cout << "Initial Population\n";
  pop.sortedPrintOn(cout);
  cout << endl;

  run_ea(ga, pop); // run the ga

  cout << "Final Population\n";
  pop.sortedPrintOn(cout);
  cout << endl;
}
