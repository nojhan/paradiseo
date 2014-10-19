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
#include <sstream>    // ostrstream, istrstream
#include <fstream>
#include <string.h>

// the general include for eo
#include <paradiseo/eo.h>

// declaration of the namespace
using namespace std;

//-----------------------------------------------------------------------------
// representation of solutions, and neighbors
#include <paradiseo/eo/ga/eoBit.h>                         // bit string : see also EO tutorial lesson 1: FirstBitGA.cpp
#include <paradiseo/mo/problems/bitString/moBitNeighbor.h> // neighbor of bit string

//-----------------------------------------------------------------------------
// fitness function, and evaluation of neighbors
#include <paradiseo/problems/eval/oneMaxEval.h>
#include <paradiseo/mo/problems/eval/moOneMaxIncrEval.h>
#include <paradiseo/mo/eval/moFullEvalByModif.h>

//-----------------------------------------------------------------------------
// neighborhood description
#include <paradiseo/mo/neighborhood/moRndWithReplNeighborhood.h> // visit one random neighbor possibly the same one several times

//-----------------------------------------------------------------------------
// the random walk local search: heuristic to sample the search space
#include <paradiseo/mo/algo/moRandomWalk.h>

//-----------------------------------------------------------------------------
// the statistics to compute during the sampling
#include <paradiseo/mo/continuator/moFitnessStat.h>
#include <paradiseo/eo/utils/eoDistance.h>
#include <paradiseo/mo/continuator/moDistanceStat.h>
#include <paradiseo/mo/continuator/moSolutionStat.h>

//-----------------------------------------------------------------------------
// the sampling class
#include <paradiseo/mo/sampling/moSampling.h>

// Declaration of types
//-----------------------------------------------------------------------------
// Indi is the typedef of the solution type like in paradisEO-eo
typedef eoBit<unsigned int> Indi;                      // bit string with unsigned fitness type
// Neighbor is the typedef of the neighbor type,
// Neighbor = How to compute the neighbor from the solution + information on it (i.e. fitness)
// all classes from paradisEO-mo use this template type
typedef moBitNeighbor<unsigned int> Neighbor ;         // bit string neighbor with unsigned fitness type


void main_function(int argc, char **argv)
{
    /* =========================================================
     *
     * Parameters
     *
     * ========================================================= */
    // more information on the input parameters: see EO tutorial lesson 3
    // but don't care at first it just read the parameters of the bit string size and the random seed.

    // First define a parser from the command-line arguments
    eoParser parser(argc, argv);

    // For each parameter, define Parameter, read it through the parser,
    // and assign the value to the variable

    // random seed parameter
    eoValueParam<uint32_t> seedParam(time(0), "seed", "Random number seed", 'S');
    parser.processParam( seedParam );
    unsigned seed = seedParam.value();

    // length of the bit string
    eoValueParam<unsigned int> vecSizeParam(20, "vecSize", "Genotype size", 'V');
    parser.processParam( vecSizeParam, "Representation" );
    unsigned vecSize = vecSizeParam.value();

    // the number of steps of the random walk
    eoValueParam<unsigned int> stepParam(100, "nbStep", "Number of steps of the random walk", 'n');
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

    // reproducible random seed: if you don't change SEED above,
    // you'll aways get the same result, NOT a random run
    // more information: see EO tutorial lesson 1 (FirstBitGA.cpp)
    rng.reseed(seed);

    /* =========================================================
     *
     * Initialization of the solution
     *
     * ========================================================= */

    // a Indi random initializer: each bit is random
    // more information: see EO tutorial lesson 1 (FirstBitGA.cpp)
    eoUniformGenerator<bool> uGen;
    eoInitFixedLength<Indi> random(vecSize, uGen);

    /* =========================================================
     *
     * Eval fitness function (full evaluation)
     *
     * ========================================================= */

    // the fitness function is just the number of 1 in the bit string
    oneMaxEval<Indi> fullEval;

    /* =========================================================
     *
     * evaluation of a neighbor solution
     *
     * ========================================================= */

    // Use it if there is no incremental evaluation: a neighbor is evaluated by the full evaluation of a solution
    // moFullEvalByModif<Neighbor> neighborEval(fullEval);

    // Incremental evaluation of the neighbor: fitness is modified by +/- 1
    moOneMaxIncrEval<Neighbor> neighborEval;

    /* =========================================================
     *
     * the neighborhood of a solution
     *
     * ========================================================= */

    // Exploration of the neighborhood in random order
    // at each step one bit is randomly generated
    moRndWithReplNeighborhood<Neighbor> neighborhood(vecSize);

    /* =========================================================
     *
     * the local search algorithm to sample the search space
     *
     * ========================================================= */

    moRandomWalk<Neighbor> walk(neighborhood, fullEval, neighborEval, nbStep);

    /* =========================================================
     *
     * the statistics to compute
     *
     * ========================================================= */

    // fitness of the solution at each step
    moFitnessStat<Indi> fStat;

    // Hamming distance to the global optimum
    eoHammingDistance<Indi> distance; // Hamming distance
    Indi bestSolution(vecSize, true); // global optimum

    moDistanceStat<Indi, double> distStat(distance, bestSolution); // statistic

    // "statistic" of the solution
    moSolutionStat<Indi> solStat;

    /* =========================================================
     *
     * The sampling of the search space
     *
     * ========================================================= */

    // sampling object :
    //    - random initialization
    //    - local search to sample the search space
    //    - one statistic to compute
    moSampling<Neighbor> sampling(random, walk, fStat);

    // to add another statistics
    sampling.add(distStat); // distance
    sampling.add(solStat); // solutions

    /* =========================================================
     *
     * execute the sampling
     *
     * ========================================================= */

    sampling();

    /* =========================================================
     *
     * export the sampling
     *
     * ========================================================= */

    // to export the statistics into file
    sampling.fileExport(str_out);

    // to get the values of statistics
    // so, you can compute some statistics in c++ from the data
    const std::vector<double> & fitnessValues = sampling.getValues(0);
    const std::vector<double> & distValues    = sampling.getValues(1);
    const std::vector<Indi>   & solutions     = sampling.getSolutions(2);

    std::cout << "First values:" << std::endl;
    std::cout << "Fitness  " << fitnessValues[0] << std::endl;
    std::cout << "Distance " << distValues[0] << std::endl;
    std::cout << "Solution " << solutions[0] << std::endl << std::endl;

    std::cout << "Last values:" << std::endl;
    std::cout << "Fitness  " << fitnessValues[fitnessValues.size() - 1] << std::endl;
    std::cout << "Distance " << distValues[distValues.size() - 1] << std::endl;
    std::cout << "Solution " << solutions[solutions.size() - 1] << std::endl;
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
