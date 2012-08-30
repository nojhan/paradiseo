//-----------------------------------------------------------------------------
/** testRandomWalk.cpp
 *
 * SV - 22/01/10
 *
 */
//-----------------------------------------------------------------------------

// standard includes
#define HAVE_SSTREAM

#include <stdexcept>  // runtime_error 
#include <iostream>   // cout
#include <sstream>  // ostrstream, istrstream
#include <fstream>
#include <string.h>

// the general include for eo
#include <eo>
#include <ga.h>

using namespace std;

//-----------------------------------------------------------------------------
// fitness function
#include <eval/oneMaxEval.h>
#include <problems/bitString/moBitNeighbor.h>
#include <eoInt.h>
#include <neighborhood/moRndWithReplNeighborhood.h>

#include <eval/moFullEvalByModif.h>
#include <eval/moFullEvalByCopy.h>
#include <continuator/moIterContinuator.h>
#include <algo/moLocalSearch.h>
#include <explorer/moRandomWalkExplorer.h>
#include <continuator/moCheckpoint.h>
#include <continuator/moFitnessStat.h>
#include <continuator/moSolutionStat.h>
#include <utils/eoDistance.h>
#include <continuator/moDistanceStat.h>

#include <utils/eoFileMonitor.h>
#include <utils/eoUpdater.h>

// REPRESENTATION
//-----------------------------------------------------------------------------
typedef eoBit<unsigned> Indi;
typedef moBitNeighbor<unsigned int> Neighbor ; // incremental evaluation
typedef moRndWithReplNeighborhood<Neighbor> Neighborhood ;

void main_function(int argc, char **argv)
{
    /* =========================================================
    *
    * Parameters
    *
    * ========================================================= */

    // First define a parser from the command-line arguments
    eoParser parser(argc, argv);

    // For each parameter, define Parameter, read it through the parser,
    // and assign the value to the variable

    eoValueParam<uint32_t> seedParam(time(0), "seed", "Random number seed", 'S');
    parser.processParam( seedParam );
    unsigned seed = seedParam.value();

    // description of genotype
    eoValueParam<unsigned int> vecSizeParam(8, "vecSize", "Genotype size", 'V');
    parser.processParam( vecSizeParam, "Representation" );
    unsigned vecSize = vecSizeParam.value();

    eoValueParam<unsigned int> stepParam(10, "nbStep", "Number of steps of the random walk", 'n');
    parser.processParam( stepParam, "Representation" );
    unsigned nbStep = stepParam.value();

    // the name of the "status" file where all actual parameter values will be saved
    string str_status = parser.ProgramName() + ".status"; // default value
    eoValueParam<string> statusParam(str_status.c_str(), "status", "Status file");
    parser.processParam( statusParam, "Persistence" );

    // do the following AFTER ALL PARAMETERS HAVE BEEN PROCESSED
    // i.e. in case you need parameters somewhere else, postpone these
    if (parser.userNeedsHelp()) {
        parser.printHelp(cout);
        exit(1);
    }
    if (statusParam.value() != "") {
        ofstream os(statusParam.value().c_str());
        os << parser;// and you can use that file as parameter file
    }

    /* =========================================================
     *
     * Random seed
     *
     * ========================================================= */

    //reproducible random seed: if you don't change SEED above,
    // you'll aways get the same result, NOT a random run
    rng.reseed(seed);


    /* =========================================================
     *
     * Eval fitness function
     *
     * ========================================================= */

    oneMaxEval<Indi> eval;


    /* =========================================================
     *
     * Initilisation of the solution
     *
     * ========================================================= */

    // a Indi random initializer
    eoUniformGenerator<bool> uGen;
    eoInitFixedLength<Indi> random(vecSize, uGen);


    /* =========================================================
     *
     * evaluation of a neighbor solution
     *
     * ========================================================= */

    moFullEvalByModif<Neighbor> nhEval(eval);

    //An eval by copy can be used instead of the eval by modif
    //moFullEvalByCopy<Neighbor> nhEval(eval);


    /* =========================================================
     *
     * the neighborhood of a solution
     *
     * ========================================================= */

    Neighborhood neighborhood(vecSize);


    /* =========================================================
     *
     * a neighborhood explorer solution
     *
     * ========================================================= */

    moRandomWalkExplorer<Neighbor> explorer(neighborhood, nhEval);


    /* =========================================================
     *
     * the continuator and the checkpoint
     *
     * ========================================================= */

    moIterContinuator<Neighbor> continuator(nbStep);

    moCheckpoint<Neighbor> checkpoint(continuator);

    moFitnessStat<Indi> fStat;
    eoHammingDistance<Indi> distance;
    Indi bestSolution(vecSize, true);
    moDistanceStat<Indi, unsigned> distStat(distance, bestSolution);
    //	moSolutionStat<Indi> solStat;

    checkpoint.add(fStat);
    checkpoint.add(distStat);
    //	checkpoint.add(solStat);

    eoValueParam<int> genCounter(-1,"Gen");
    eoIncrementor<int> increm(genCounter.value());
    checkpoint.add(increm);

    eoFileMonitor outputfile("out.dat", " ");
    checkpoint.add(outputfile);

    outputfile.add(genCounter);
    outputfile.add(fStat);
    outputfile.add(distStat);
    //	outputfile.add(solStat);

    Indi solution; // current solution of the search process

    /*
    // to save the solution at each iteration
    eoState outState;

    // Register the algorithm into the state (so it has something to save!!

    outState.registerObject(solution);

    // and feed the state to state savers
    // save state every 10th iteration
    eoCountedStateSaver stateSaver(10, outState, "iteration");

      // Don't forget to add the two savers to the checkpoint
    checkpoint.add(stateSaver);
    */

    /* =========================================================
     *
     * the local search algorithm
     *
     * ========================================================= */

    moLocalSearch<Neighbor> localSearch(explorer, checkpoint, eval);

    /* =========================================================
     *
     * execute the local search from random sollution
     *
     * ========================================================= */

    random(solution);

    //Can be eval here, else it will be done at the beginning of the localSearch
    //eval(solution);

    std::cout << "initial: " << solution << std::endl ;

    localSearch(solution);

    std::cout << "final:   " << solution << std::endl ;

}

// A main that catches the exceptions

int main(int argc, char **argv)
{
    try {
        main_function(argc, argv);
    }
    catch (exception& e) {
        cout << "Exception: " << e.what() << '\n';
    }
    return 1;
}
