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
#include <es.h>

#include "real_value.h"		// the sphere fitness

// Now the main 
/////////////// 
typedef eoMinimizingFitness  FitT;

template <class EOT>
void runAlgorithm(EOT, eoParser& _parser, eoState& _state, eoRealVectorBounds& _bounds, eoValueParam<string> _load_name);
  
int main(int argc, char *argv[]) 
{
  // Create the command-line parser
  eoParser parser( argc, argv, "Basic EA for vector<float> with adaptive mutations");

  // Define Parameters and load them
  eoValueParam<uint32>& seed        = parser.createParam(static_cast<uint32>(time(0)), "seed", "Random number seed");
  eoValueParam<string>& load_name   = parser.createParam(string(), "Load","Load a state file",'L');
  eoValueParam<string>& save_name   = parser.createParam(string(), "Save","Saves a state file",'S');
  eoValueParam<bool>&   stdevs      = parser.createParam(false, "Stdev", "Use adaptive mutation rates", 's');
  eoValueParam<bool>&   corr        = parser.createParam(false, "Correl", "Use correlated mutations", 'c');
  eoValueParam<unsigned>& chromSize = parser.createParam(unsigned(50), "ChromSize", "Number of chromosomes", 'n');
  eoValueParam<double>& minimum     = parser.createParam(-1.0, "Min", "Minimum for Objective Variables", 'l');
  eoValueParam<double>& maximum     = parser.createParam(1.0, "Max", "Maximum for Objective Variables", 'h');

  eoState state;
    state.registerObject(parser);
    rng.reseed(seed.value());

   if (!load_name.value().empty())
   { // load the parser. This is only neccessary when the user wants to 
     // be able to change the parameters in the state file by hand
     // Note that only parameters inserted in the parser at this point
     // will be loaded!.
       state.load(load_name.value()); // load the parser
   }

    state.registerObject(rng);

    eoRealVectorBounds bounds(chromSize.value(), minimum.value(), maximum.value());
    
    // Run the appropriate algorithm
    if (stdevs.value() == false && corr.value() == false)
    {
        runAlgorithm(eoEsSimple<FitT>() ,parser, state, bounds, load_name);
    }
    else if (corr.value() == true)
    {
        runAlgorithm(eoEsFull<FitT>(),parser, state, bounds, load_name);
    }
    else 
    {
        runAlgorithm(eoEsStdev<FitT>(), parser, state, bounds, load_name);
    }

    // and save
    if (!save_name.value().empty())
    {
        string file_name = save_name.value();
        save_name.value() = ""; // so that it does not appear in the parser section of the state file
        state.save(file_name);
    }

	return 0;  
}

template <class EOT>
void runAlgorithm(EOT, eoParser& _parser, eoState& _state, eoRealVectorBounds& _bounds, eoValueParam<string> _load_name)
{
    // evaluation
    eoEvalFuncPtr<EOT, double, const vector<double>&> eval(  real_value );

    // population parameters, unfortunately these can not be altered in the state file
    eoValueParam<unsigned> mu = _parser.createParam(unsigned(50), "mu","Size of the population");
    eoValueParam<float>lambda_rate = _parser.createParam(float(7.0), "lambda_rate", "Factor of children to produce");

    if (lambda_rate.value() < 1.0f)
    {
        throw logic_error("lambda_rate must be larger than 1 in a comma strategy");
    }

    // Initialization
    eoEsChromInit<EOT> init(_bounds);
    
    // State takes ownership of pop because it needs to save it in caller
    eoPop<EOT>& pop = _state.takeOwnership(eoPop<EOT>(mu.value(), init));

    _state.registerObject(pop);

    if (!_load_name.value().empty())
    { // The real loading happens here when all objects are registered
       _state.load(_load_name.value()); // load all and everything
    }
    else
    {
        // evaluate initial population
        apply<EOT>(eval, pop);
    }

    // Ok, time to set up the algorithm
    // Proxy for the mutation parameters
    eoEsMutationInit mutateInit(_parser);

    eoEsMutate<EOT> mutate(mutateInit, _bounds);

    // monitoring, statistics etc.
    eoAverageStat<EOT> average;
    eoStdoutMonitor monitor;

    monitor.add(average);

    eoGenContinue<EOT> cnt(100);
    eoCheckPoint<EOT> checkpoint(cnt);
    checkpoint.add(monitor);
    checkpoint.add(average);


    eoProportionalGOpSel<EOT> opSel;
    opSel.addOp(mutate, 1.0);

    eoEvolutionStrategy<EOT> es(checkpoint, eval, opSel, lambda_rate.value(), eoEvolutionStrategy<EOT>::comma_strategy());

    es(pop);

    pop.sort();
    cout << "Final population\n" << pop << endl;

}
