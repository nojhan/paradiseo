//-----------------------------------------------------------------------------
// gprop
//-----------------------------------------------------------------------------

#include <stdlib.h>                // EXIT_SUCCESS EXIT_FAILURE
#include <stdexcept>               // exception 
#include <iostream>                // cerr cout
#include <fstream>                 // ifstream
#include <string>                  // string
#include <utils/eoParser.h>        // eoParser
#include <eoPop.h>                 // eoPop
#include <eoEvalFuncPtr.h>         // eoEvalFunc
#include <eoStochTournament.h>     // eoStochTournament
#include <eoGenContinue.h>         // eoGenContinue
#include <eoFitContinue.h>         // eoFitContinue
#include <eoCombinedContinue.h>    // eoCombinedContinue
#include <utils/eoCheckPoint.h>    // eoCheckPoint
#include <utils/eoStat.h>          // eoBestFitnessStat
#include <utils/eoStdoutMonitor.h> // eoStdoutMonitor
#include <eoSGA.h>                 // eoSGA
#include "gprop.h"                 // Chrom eoChromInit eoChromMutation eoChromXover eoChromEvaluator

//-----------------------------------------------------------------------------
// global variables
//-----------------------------------------------------------------------------

unsigned in, out, hidden;

//-----------------------------------------------------------------------------
// parameters
//-----------------------------------------------------------------------------

eoValueParam<unsigned> pop_size(10, "pop_size", "population size", 'p');
eoValueParam<unsigned> generations(10, "generations", "number of generation", 'g');
eoValueParam<double> mut_rate(0.25, "mut_rate", "mutation rate", 'm');
eoValueParam<double> xover_rate(0.25, "xover_rate", "default crossover rate", 'x');
eoValueParam<string> file("", "file", "common start of patterns filenames *.trn *.val and *.tst", 'f');
eoValueParam<unsigned> hiddenp(0, "hidden", "number of neurons in hidden layer", 'd');

//-----------------------------------------------------------------------------
// auxiliar functions
//-----------------------------------------------------------------------------

void arg(int argc, char** argv);
void load_file(mlp::set& s, const string& s);
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
  parser.processParam(file,        "files");
  parser.processParam(hiddenp,     "genetic operators");

  if (parser.userNeedsHelp())
    {
      parser.printHelp(cout);
      exit(EXIT_SUCCESS);
    }

  load_file(trn_set, "trn");
  load_file(val_set, "val");
  load_file(tst_set, "tst");
    
  phenotype::trn_max = trn_set.size();
  phenotype::val_max = val_set.size();
  phenotype::tst_max = tst_set.size();

  in = trn_set.front().input.size();
  out = trn_set.front().output.size();
  hidden = hiddenp.value();
}

//-----------------------------------------------------------------------------

void load_file(mlp::set& set, const string& ext)
{
  string filename = file.value(); filename += "." + ext;

  ifstream ifs(filename.c_str());
  if (!ifs)
    {
      cerr << "can't open file \"" << filename << "\"" << endl;
      exit(EXIT_FAILURE);
    } 
  
  ifs >> set;

  if (set.size() == 0)
    {
      cerr << filename << " data file is empty!";
      exit(EXIT_FAILURE);
    }
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
  eoStochTournament<Chrom> select;

  // genetic operators
  eoChromMutation mutation;
  eoChromXover xover;
  
  // stop condition
  eoGenContinue<Chrom> continuator1(generations.value());
  phenotype p; p.val_ok = val_set.size() - 1; p.mse_error = 0;
  eoFitContinue<Chrom> continuator2(p);
  eoCombinedContinue<Chrom> continuator(continuator1, continuator2);

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

  cout << "best: " << *max_element(pop.begin(), pop.end()) << endl;
}

//-----------------------------------------------------------------------------

// Local Variables: 
// mode:C++
// End:
