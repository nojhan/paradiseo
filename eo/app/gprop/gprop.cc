//-----------------------------------------------------------------------------
// gprop
//-----------------------------------------------------------------------------

#include <stdlib.h>         // EXIT_SUCCESS EXIT_FAILURE
#include <stdexcept>        // exception 
#include <iostream>         // cerr cout
#include <fstream>          // ifstream
#include <string>           // string
#include <utils/eoParser.h> // eoParser
#include <eoPop.h>          // eoPop
#include <eoGenContinue.h>  // eoGenContinue
#include <eoProportional.h> // eoProportional
#include <eoSGA.h>          // eoSGA
#include "gprop.h"          // Chrom eoChromInit eoChromMutation eoChromXover eoChromEvaluator

//-----------------------------------------------------------------------------
// global variables
//-----------------------------------------------------------------------------

unsigned in, out, hidden;

//-----------------------------------------------------------------------------
// parameters
//-----------------------------------------------------------------------------

eoValueParam<unsigned> pop_size(10, "pop_size", "default population size", 'p');
eoValueParam<unsigned> generations(10, "generations", "default generation number", 'g');
eoValueParam<double> mut_rate(0.1, "mut_rate", "default mutation rate", 'm');
eoValueParam<double> xover_rate(0.1, "xover_rate", "default crossover rate", 'x');
eoValueParam<string> file("", "file", "common part of patterns filenames *.trn *.val and *.tst", 'f');
eoValueParam<unsigned> hiddenp(8, "hidden", "default number of neurons in hidden layer", 'h');

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

  cout << "set.size() = " << set.size() << endl;
  
  if (set.size() == 0)
    {
      cerr << filename << " data file is empty!";
      exit(EXIT_FAILURE);
    }
}

//-----------------------------------------------------------------------------

void ga()
{
  eoGenContinue<Chrom> continuator(generations.value());
 
  eoProportional<Chrom> select;
  eoChromMutation mutation(generations);
  eoChromXover xover;
  eoEvalFuncPtr<Chrom> evaluator(eoChromEvaluator);

  eoSGA<Chrom> sga(select,
		   xover, xover_rate.value(), 
		   mutation, mut_rate.value(), 
		   evaluator, 
		   continuator);

  eoInitChrom init;
  eoPop<Chrom> pop(pop_size.value(), init);
  apply<Chrom>(evaluator, pop);

  cout << pop << endl;
  
  sga(pop);

  cout << pop << endl;
}

//-----------------------------------------------------------------------------

// Local Variables: 
// mode:C++
// End:
