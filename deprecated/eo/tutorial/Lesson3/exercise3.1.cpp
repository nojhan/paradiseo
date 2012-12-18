//-----------------------------------------------------------------------------
// SecondBitGA.cpp
//-----------------------------------------------------------------------------
//*
// Same code than FirstBitEA as far as Evolutionary Computation is concerned
// but now you learn to enter the parameters in a more flexible way
// and to twidle the output to your preferences!
//-----------------------------------------------------------------------------
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

// standard includes
#include <fstream>
#include <iostream>   // cout
#include <stdexcept>  // runtime_error

// the general include for eo
#include <eo>

// EVAL
#include "binary_value.h"

// REPRESENTATION
//-----------------------------------------------------------------------------
// Include the corresponding file
#include <ga.h>		 // bitstring representation & operators
// define your genotype and fitness types
typedef eoBit<eoMinimizingFitness> Indi;

using namespace std;

// the main_function: nothing changed(!), except variable initialization
void main_function(int argc, char **argv)
{
// PARAMETRES
//-----------------------------------------------------------------------------
// instead of having all values of useful parameters as constants, read them:
// either on the command line (--option=value or -o=value)
//     or in a parameter file (same syntax, order independent,
//                             # = usual comment character
//     or in the environment (TODO)

  // First define a parser from the command-line arguments
    eoParser parser(argc, argv);

    // For each parameter, define Parameter, read it through the parser,
    // and assign the value to the variable

    eoValueParam<uint32_t> seedParam(time(0), "seed", "Random number seed", 'S');
    parser.processParam( seedParam );
    unsigned seed = seedParam.value();

   // description of genotype
    eoValueParam<unsigned int> vecSizeParam(100, "vecSize", "Genotype size",'V');
    parser.processParam( vecSizeParam, "Representation" );
    unsigned vecSize = vecSizeParam.value();

   // parameters for evolution engine
    eoValueParam<unsigned int> popSizeParam(100, "popSize", "Population size",'P');
    parser.processParam( popSizeParam, "Evolution engine" );
    unsigned popSize = popSizeParam.value();

    eoValueParam<unsigned int> tSizeParam(10, "tSize", "Tournament size",'T');
    parser.processParam( tSizeParam, "Evolution Engine" );
    unsigned tSize = tSizeParam.value();

   // init and stop
    eoValueParam<string> loadNameParam("", "Load","A save file to restart from",'L');
    parser.processParam( loadNameParam, "Persistence" );
    string loadName = loadNameParam.value();

    eoValueParam<unsigned int> maxGenParam(500, "maxGen", "Maximum number of generations",'G');
    parser.processParam( maxGenParam, "Stopping criterion" );
    unsigned maxGen = maxGenParam.value();

    eoValueParam<unsigned int> minGenParam(500, "minGen", "Minimum number of generations",'g');
    parser.processParam( minGenParam, "Stopping criterion" );
    unsigned minGen = minGenParam.value();

    eoValueParam<unsigned int> steadyGenParam(100, "steadyGen", "Number of generations with no improvement",'s');
    parser.processParam( steadyGenParam, "Stopping criterion" );
    unsigned steadyGen = steadyGenParam.value();

   // operators probabilities at the algorithm level
    eoValueParam<double> pCrossParam(0.6, "pCross", "Probability of Crossover", 'C');
    parser.processParam( pCrossParam, "Genetic Operators" );
    double pCross = pCrossParam.value();

    eoValueParam<double> pMutParam(0.1, "pMut", "Probability of Mutation", 'M');
    parser.processParam( pMutParam, "Genetic Operators" );
    double pMut = pMutParam.value();

   // relative rates for crossovers
    eoValueParam<double> onePointRateParam(1, "onePointRate", "Relative rate for one point crossover", '1');
    parser.processParam( onePointRateParam, "Genetic Operators" );
    double onePointRate = onePointRateParam.value();

    eoValueParam<double> twoPointsRateParam(1, "twoPointRate", "Relative rate for two point crossover", '2');
    parser.processParam( twoPointsRateParam, "Genetic Operators" );
    double twoPointsRate = twoPointsRateParam.value();

    eoValueParam<double> uRateParam(2, "uRate", "Relative rate for uniform crossover", 'U');
    parser.processParam( uRateParam, "Genetic Operators" );
    double URate =  uRateParam.value();

   // relative rates and private parameters for mutations;
    eoValueParam<double> pMutPerBitParam(0.01, "pMutPerBit", "Probability of flipping 1 bit in bit-flip mutation", 'b');
    parser.processParam( pMutPerBitParam, "Genetic Operators" );
    double pMutPerBit = pMutPerBitParam.value();

    eoValueParam<double> bitFlipRateParam(0.01, "bitFlipRate", "Relative rate for bit-flip mutation", 'B');
    parser.processParam( bitFlipRateParam, "Genetic Operators" );
    double bitFlipRate =  bitFlipRateParam.value();

    eoValueParam<double> oneBitRateParam(0.01, "oneBitRate", "Relative rate for deterministic bit-flip mutation", 'D');
    parser.processParam( oneBitRateParam, "Genetic Operators" );
    double oneBitRate = oneBitRateParam.value();

    // the name of the "status" file where all actual parameter values will be saved
    string str_status = parser.ProgramName() + ".status"; // default value
    eoValueParam<string> statusParam(str_status.c_str(), "status","Status file",'S');
    parser.processParam( statusParam, "Persistence" );

   // do the following AFTER ALL PARAMETERS HAVE BEEN PROCESSED
   // i.e. in case you need parameters somewhere else, postpone these
    if (parser.userNeedsHelp())
      {
	parser.printHelp(cout);
	exit(1);
      }
    if (statusParam.value() != "")
      {
	ofstream os(statusParam.value().c_str());
	os << parser;		// and you can use that file as parameter file
      }

// EVAL
  /////////////////////////////
  // Fitness function
  ////////////////////////////
  // Evaluation: from a plain C++ fn to an EvalFunc Object ...
  eoEvalFuncPtr<Indi, double, const vector<bool>& > plainEval( binary_value );
  // ... to an object that counts the nb of actual evaluations
  eoEvalFuncCounter<Indi> eval(plainEval);

// INIT
  ////////////////////////////////
  // Initilisation of population
  ////////////////////////////////
  // Either load or initialize
  // create an empty pop
  eoPop<Indi> pop;
  // create a state for reading
  eoState inState;		// a state for loading - WITHOUT the parser
  // register the rng and the pop in the state, so they can be loaded,
  // and the present run will be the exact conitnuation of the saved run
  // eventually with different parameters
  inState.registerObject(rng);
  inState.registerObject(pop);

  if (loadName != "")
    {
      inState.load(loadName); //  load the pop and the rng
      // the fitness is read in the file:
      // do only evaluate the pop if the fitness has changed
    }
  else
    {
      rng.reseed(seed);
      // a Indi random initializer
      // based on boolean_generator class (see utils/rnd_generator.h)
      eoUniformGenerator<bool> uGen;
      eoInitFixedLength<Indi> random(vecSize, uGen);

      // Init pop from the randomizer: need to use the append function
      pop.append(popSize, random);
      // and evaluate pop (STL syntax)
      apply<Indi>(eval, pop);
    } // end of initializatio of the population

// OUTPUT
  // sort pop for pretty printout
  //   pop.sort();
  // Print (sorted) intial population (raw printout)
  cout << "Initial Population" << endl << pop ;
  cout << "and best is " << pop.best_element() << "\n\n";
  cout << "and worse is " << pop.worse_element() << "\n\n";
// ENGINE
  /////////////////////////////////////
  // selection and replacement
  ////////////////////////////////////
// SELECT
  // The robust tournament selection
  eoDetTournamentSelect<Indi> selectOne(tSize);       // tSize in [2,POPSIZE]
  // is now encapsulated in a eoSelectPerc (entage)
  eoSelectPerc<Indi> select(selectOne);// by default rate==1

// REPLACE
  // And we now have the full slection/replacement - though with
  // generational replacement at the moment :-)
  eoGenerationalReplacement<Indi> replace;
  // want to add (weak) elitism? easy!
  // rename the eoGenerationalReplacement replace_main,
  // then encapsulate it in the elitist replacement
  //  eoWeakElitistReplacement<Indi> replace(replace_main);

// OPERATORS
  //////////////////////////////////////
  // The variation operators
  //////////////////////////////////////
// CROSSOVER
  // 1-point crossover for bitstring
  eo1PtBitXover<Indi> xover1;
  // uniform crossover for bitstring
  eoUBitXover<Indi> xoverU;
  // 2-pots xover
  eoNPtsBitXover<Indi> xover2(2);
  // Combine them with relative rates
  eoPropCombinedQuadOp<Indi> xover(xover1, onePointRate);
  xover.add(xoverU, URate);
  xover.add(xover2, twoPointsRate, true);

// MUTATION
  // standard bit-flip mutation for bitstring
  eoBitMutation<Indi>  mutationBitFlip(pMutPerBit);
  // mutate exactly 1 bit per individual
  eoDetBitFlip<Indi> mutationOneBit;
  // Combine them with relative rates
  eoPropCombinedMonOp<Indi> mutation(mutationBitFlip, bitFlipRate);
  mutation.add(mutationOneBit, oneBitRate, true);

  // The operators are  encapsulated into an eoTRansform object
  eoSGATransform<Indi> transform(xover, pCross, mutation, pMut);

// STOP
  //////////////////////////////////////
  // termination condition see FirstBitEA.cpp
  /////////////////////////////////////
  eoGenContinue<Indi> genCont(maxGen);
  eoSteadyFitContinue<Indi> steadyCont(minGen, steadyGen);
  // eoFitContinue<Indi> fitCont(vecSize);   // remove if minimizing :-)
  eoCombinedContinue<Indi> continuator(genCont);
  continuator.add(steadyCont);
  //  continuator.add(fitCont);
  // Ctrl C signal handling: don't know if that works in MSC ...
#ifndef _MSC_VER
  eoCtrlCContinue<Indi> ctrlC;
  continuator.add(ctrlC);
#endif

// CHECKPOINT
  // but now you want to make many different things every generation
  // (e.g. statistics, plots, ...).
  // the class eoCheckPoint is dedicated to just that:

  // Declare a checkpoint (from a continuator: an eoCheckPoint
  // IS AN eoContinue and will be called in the loop of all algorithms)
  eoCheckPoint<Indi> checkpoint(continuator);

    // Create a counter parameter
    eoValueParam<unsigned> generationCounter(0, "Gen.");

    // Create an incrementor (sub-class of eoUpdater). Note that the
    // parameter's value is passed by reference,
    // so every time the incrementer is updated (every generation),
    // the data in generationCounter will change.
    eoIncrementor<unsigned> increment(generationCounter.value());

    // Add it to the checkpoint,
    // so the counter is updated (here, incremented) every generation
    checkpoint.add(increment);

    // now some statistics on the population:
    // Best fitness in population
    eoBestFitnessStat<Indi> bestStat;
    eoAverageStat<Indi> averageStat;
    // Second moment stats: average and stdev
    eoSecondMomentStats<Indi> SecondStat;
    // the Fitness Distance Correlation
    // need first an object to compute the distances
    eoQuadDistance<Indi> dist;	// Hamming distance
    eoFDCStat<Indi> fdcStat(dist);

    // Add them to the checkpoint to get them called at the appropriate time
    checkpoint.add(bestStat);
    checkpoint.add(averageStat);
    checkpoint.add(SecondStat);
    checkpoint.add(fdcStat);

    // The Stdout monitor will print parameters to the screen ...
    eoStdoutMonitor monitor(false);

    // when called by the checkpoint (i.e. at every generation)
    checkpoint.add(monitor);

    // the monitor will output a series of parameters: add them
    monitor.add(generationCounter);
    monitor.add(eval);		// because now eval is an eoEvalFuncCounter!
    monitor.add(bestStat);
    monitor.add(SecondStat);
    monitor.add(fdcStat);

    // test de eoPopStat and/or eoSortedPopStat.
    // Dumps the whole pop every 10 gen.
    //    eoSortedPopStat<Indi> popStat(10, "Dump of whole population");
//     eoPopStat<Indi> popStat(10, "Dump of whole population");
//     checkpoint.add(popStat);
//     monitor.add(popStat);

    // A file monitor: will print parameters to ... a File, yes, you got it!
    eoFileMonitor fileMonitor("stats.xg", " ");

    // the checkpoint mechanism can handle monitors
    checkpoint.add(fileMonitor);

    // the fileMonitor can monitor parameters, too, but you must tell it!
    fileMonitor.add(generationCounter);
    fileMonitor.add(bestStat);
    fileMonitor.add(SecondStat);

#ifndef _MSC_VER
    // and an eoGnuplot1DMonitor will 1-print to a file, and 2- plot on screen
    eoGnuplot1DMonitor gnuMonitor("best_average.xg",minimizing_fitness<Indi>());
    // the checkpoint mechanism can handle multiple monitors
    checkpoint.add(gnuMonitor);
    // the gnuMonitor can monitor parameters, too, but you must tell it!
    gnuMonitor.add(eval);
    gnuMonitor.add(bestStat);
    gnuMonitor.add(averageStat);

    // send a scaling command to gnuplot
    gnuMonitor.gnuplotCommand("set yrange [0:500]");

    // a specific plot monitor for FDC
    // first into a file (it adds everything ti itself
    eoFDCFileSnapshot<Indi> fdcFileSnapshot(fdcStat);
    // then to a Gnuplot monitor
    eoGnuplot1DSnapshot fdcGnuplot(fdcFileSnapshot);
    // and of coruse add them to the checkPoint
    checkpoint.add(fdcFileSnapshot);
    checkpoint.add(fdcGnuplot);

    // want to see how the fitness is spread?
    eoScalarFitnessStat<Indi> fitStat;
    checkpoint.add(fitStat);
    // a gnuplot-based monitor for snapshots: needs a dir name
    // where to store the files
    eoGnuplot1DSnapshot fitSnapshot("Fitnesses");
    // add any stat that is a vector<double> to it
    fitSnapshot.add(fitStat);
    // and of course add it to the checkpoint
    checkpoint.add(fitSnapshot);
#endif
    // Last type of item the eoCheckpoint can handle: state savers:
    eoState outState;
    // Register the algorithm into the state (so it has something to save!!)
    outState.registerObject(rng);
    outState.registerObject(pop);

    // and feed the state to state savers
    // save state every 100th  generation
    eoCountedStateSaver stateSaver1(100, outState, "generation");
    // save state every 1 seconds
    eoTimedStateSaver   stateSaver2(1, outState, "time");

    // Don't forget to add the two savers to the checkpoint
    checkpoint.add(stateSaver1);
    checkpoint.add(stateSaver2);
    // and that's it for the (control and) output

// GENERATION
  /////////////////////////////////////////
  // the algorithm
  ////////////////////////////////////////

  // Easy EA requires
  // selection, transformation, eval, replacement, and stopping criterion
  eoEasyEA<Indi> gga(checkpoint, eval, select, transform, replace);

  // Apply algo to pop - that's it!
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
