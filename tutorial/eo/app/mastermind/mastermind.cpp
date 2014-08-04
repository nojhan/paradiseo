//-----------------------------------------------------------------------------
// mastermind
//-----------------------------------------------------------------------------

#include <stdlib.h>                // EXIT_SUCCESS EXIT_FAILURE
#include <stdexcept>               // exception
#include <iostream>                // cerr cout
#include <fstream>                 // ifstream
#include <string>                  // string
#include <paradiseo/eo.h>          // all usefull eo stuff

#include "mastermind.h"            // Chrom eoChromInit eoChromMutation eoChromXover eoChromEvaluator

using namespace std;

//-----------------------------------------------------------------------------
// global variables
//-----------------------------------------------------------------------------

unsigned in, out, hidden;

//-----------------------------------------------------------------------------
// parameters
//-----------------------------------------------------------------------------

eoValueParam<unsigned> pop_size(16, "pop_size", "population size", 'p');
eoValueParam<unsigned> generations(100, "generations", "number of generation", 'g');
eoValueParam<double> mut_rate(0.1, "mut_rate", "mutation rate", 'm');
eoValueParam<double> xover_rate(0.5, "xover_rate", "default crossover rate", 'x');
eoValueParam<unsigned> col_p(default_colors, "colors", "number of colors", 'c');
eoValueParam<unsigned> len_p(default_length, "legth", "solution legth", 'l');
eoValueParam<string> sol_p(default_solution, "solution", "problem solution", 's');

//-----------------------------------------------------------------------------
// auxiliar functions
//-----------------------------------------------------------------------------

void arg(int argc, char** argv);
void ga();

//-----------------------------------------------------------------------------
// main
//-----------------------------------------------------------------------------

int main(int argc, char** argv)
{
  try
    {
      arg(argc, argv);
      ga();
    }
  catch (exception& e)
    {
	cerr << argv[0] << ": " << e.what() << endl;
	exit(EXIT_FAILURE);
    }

  return 0;
}

//-----------------------------------------------------------------------------
// implementation
//-----------------------------------------------------------------------------

void arg(int argc, char** argv)
{
  eoParser parser(argc, argv);

  parser.processParam(pop_size,    "genetic operators");
  parser.processParam(generations, "genetic operators");
  parser.processParam(mut_rate,    "genetic operators");
  parser.processParam(xover_rate,  "genetic operators");
  parser.processParam(col_p,       "problem");
  parser.processParam(len_p,       "problem");
  parser.processParam(sol_p,       "problem");

  if (parser.userNeedsHelp())
    {
      parser.printHelp(cout);
      exit(EXIT_SUCCESS);
    }

  init_eoChromEvaluator(col_p.value(), len_p.value(), sol_p.value());
}

//-----------------------------------------------------------------------------

void ga()
{
  // create population
  eoInitChrom init;
  eoPop<Chrom> pop(pop_size.value(), init);

  // evaluate population
  eoEvalFuncPtr<Chrom> evaluator(eoChromEvaluator);
  apply<Chrom>(evaluator, pop);

  // selector
  eoProportionalSelect<Chrom> select(pop);

  // genetic operators
  eoChromMutation mutation;
  eoChromXover xover;

  // stop condition
  eoGenContinue<Chrom> continuator1(generations.value());
  eoFitContinue<Chrom> continuator2(solution.fitness());
  eoCombinedContinue<Chrom> continuator(continuator1);
  continuator.add(continuator2);

  // checkpoint
  eoCheckPoint<Chrom> checkpoint(continuator);

  // monitor
  eoStdoutMonitor monitor;
  checkpoint.add(monitor);

  // statistics
  eoBestFitnessStat<Chrom> stats;
  checkpoint.add(stats);
  monitor.add(stats);

  // genetic algorithm
  eoSGA<Chrom> sga(select,
		   xover, xover_rate.value(),
		   mutation, mut_rate.value(),
		   evaluator,
		   checkpoint);
  sga(pop);

  cout << "solution = " << solution << endl
       << "best     = " << *max_element(pop.begin(), pop.end()) << endl;
}

//-----------------------------------------------------------------------------

// Local Variables:
// mode:C++
// End:
