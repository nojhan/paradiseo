//-----------------------------------------------------------------------------
/** sampling.cpp
 *
 * SV - 03/05/10
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
#include <continuator/moTrueContinuator.h>
#include <algo/moLocalSearch.h>
#include <explorer/moRandomWalkExplorer.h>
#include <continuator/moCheckpoint.h>
#include <continuator/moFitnessStat.h>
#include <continuator/moSolutionStat.h>
#include <utils/eoDistance.h>
#include <continuator/moDistanceStat.h>

#include <utils/eoFileMonitor.h>
#include <utils/eoUpdater.h>

#include <sampling/moSampling.h>

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

    // the name of the output file
    string str_out = "out.dat"; // default value
    eoValueParam<string> outParam(str_out.c_str(), "out", "Output file of the sampling", 'o');
    parser.processParam(outParam, "Persistence" );

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

    moRandomWalkExplorer<Neighbor> explorer(neighborhood, nhEval, nbStep);


    /* =========================================================
     *
     * the continuator and the checkpoint
     *
     * ========================================================= */

    moTrueContinuator<Neighbor> continuator;//always continue

    moFitnessStat<Indi, unsigned> fStat;
    eoHammingDistance<Indi> distance;
    Indi bestSolution(vecSize, true);
    moDistanceStat<Indi, unsigned> distStat(distance, bestSolution);

    /* =========================================================
     *
     * the local search algorithm
     *
     * ========================================================= */

    moLocalSearch<Neighbor> localSearch(explorer, continuator, eval);

    /* =========================================================
     *
     * The sampling of the search space
     *
     * ========================================================= */
    
    moSampling<Neighbor> sampling(random, localSearch, fStat);

    /* =========================================================
     *
     * execute the sampling
     *
     * ========================================================= */

    sampling();

    sampling.exportFile(str_out);

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
