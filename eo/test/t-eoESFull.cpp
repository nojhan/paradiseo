// Program to test several EO-ES features

#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif 

#include <algorithm>
#include <string>
#include <iostream>
#include <iterator>
#include <stdexcept>

using namespace std;

#include <utils/eoParser.h>
#include <utils/eoState.h>

#include <utils/eoStat.h>
#include <utils/eoFileMonitor.h>

// population
#include <eoPop.h>

// evaluation specific
#include <eoEvalFuncPtr.h>

// representation specific
#include <es/evolution_strategies>

#include "real_value.h"		// the sphere fitness

// Now the main 
/////////////// 
typedef double  FitT;

template <class EOT>
void runAlgorithm(EOT, eoParser& _parser, eoEsObjectiveBounds& _bounds);
  
main(int argc, char *argv[]) 
{
  // Create the command-line parser
  eoParser parser( argc, argv, "Basic EA for vector<float> with adaptive mutations");

  // Define Parameters and load them
  eoValueParam<uint32>& seed        = parser.createParam(static_cast<uint32>(time(0)), "seed", "Random number seed");
  eoValueParam<string>& load_name   = parser.createParam(string(), "Load","Load a state file",'L');
  eoValueParam<string>& save_name   = parser.createParam(string(), "Save","Saves a state file",'S');
  eoValueParam<bool>&   stdevs      = parser.createParam(true, "Stdev", "Use adaptive mutation rates", 's');
  eoValueParam<bool>&   corr        = parser.createParam(true, "Correl", "Use correlated mutations", 'c');
  eoValueParam<unsigned>& chromSize = parser.createParam(unsigned(1), "ChromSize", "Number of chromosomes", 'n');
  eoValueParam<double>& minimum     = parser.createParam(-1.e5, "Min", "Minimum for Objective Variables", 'l');
  eoValueParam<double>& maximum     = parser.createParam(1.e5, "Max", "Maximum for Objective Variables", 'h');

  eoState state;
    state.registerObject(parser);
    rng.reseed(seed.value());

   if (!load_name.value().empty())
   { // load the parser. This is only neccessary when the user wants to 
     // be able to change the parameters in the state file by hand.
       state.load(load_name.value()); // load the parser
   }

    state.registerObject(rng);

    eoEsObjectiveBounds bounds(chromSize.value(), minimum.value(), maximum.value());
    
    // Run the appropriate algorithm
    if (stdevs.value() == false && corr.value() == false)
    {
        runAlgorithm(eoEsSimple<FitT>() ,parser, bounds);
    }
    else if (corr.value() == true)
    {
        runAlgorithm(eoEsFull<FitT>(),parser, bounds);
    }
    else 
    {
        runAlgorithm(eoEsStdev<FitT>(), parser, bounds);
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
void runAlgorithm(EOT, eoParser& _parser, eoEsObjectiveBounds& _bounds)
{
    // evaluation
    eoEvalFuncPtr<eoFixedLength<FitT, double> > eval(  real_value );

    // population parameters, unfortunately these can not be altered in the state file
    eoValueParam<unsigned> mu = _parser.createParam(unsigned(50), "mu","Size of the population");
    eoValueParam<unsigned>lambda = _parser.createParam(unsigned(250), "lambda", "No. of children to produce");

    if (mu.value() > lambda.value())
    {
        throw logic_error("Mu must be smaller than lambda in a comma strategy");
    }

    // Initialization
    eoEsChromInit<EOT> init(_bounds);
    eoPop<EOT> pop(mu.value(), init);

    // evaluate initial population
    eval.range(pop.begin(), pop.end());
    
    // Ok, time to set up the algorithm
    // Proxy for the mutation parameters
    eoEsMutationInit mutateInit(_parser);

    eoEsMutate<EOT> mutate(mutateInit, _bounds);

    // monitoring, statistics etc.
    eoAverageStat<EOT> average;
    eoFileMonitor monitor("test.csv");

    monitor.add(average);

    // Okok, I'm lazy, here's the algorithm defined inline

    for (unsigned i = 0; i < 20; ++i)
    {
        pop.resize(pop.size() + lambda.value());

        for (unsigned j = mu.value(); j < pop.size(); ++j)
        {
            pop[j] = pop[rng.random(mu.value())];
            mutate(pop[j]);
            eval(pop[j]);
        }

        // comma strategy
        std::sort(pop.begin() + mu.value(), pop.end());
        copy(pop.begin() + mu.value(), pop.begin() + 2 * mu.value(), pop.begin());
        pop.resize(mu.value());
    
        average(pop);
        monitor();
    }
}
