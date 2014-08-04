//-----------------------------------------------------------------------------
/** neutralWalk.cpp
 *
 * SV - 07/05/10
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
#include <paradiseo/problems/eval/royalRoadEval.h>
#include <paradiseo/mo/problems/eval/moRoyalRoadIncrEval.h>

//-----------------------------------------------------------------------------
// neighborhood description
#include <paradiseo/mo/neighborhood/moRndWithoutReplNeighborhood.h> // visit one random neighbor possibly the same one several times

//-----------------------------------------------------------------------------
// the sampling class
#include <paradiseo/eo/utils/eoDistance.h>
#include <paradiseo/mo/sampling/moNeutralWalkSampling.h>

//-----------------------------------------------------------------------------
// the statistics class
#include <paradiseo/mo/sampling/moStatistics.h>

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

    // size of the block
    eoValueParam<unsigned int> blockSizeParam(4, "blockSize", "Block size of the Royal Road", 'k');
    parser.processParam( blockSizeParam, "Representation" );
    unsigned blockSize = blockSizeParam.value();

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
     * Eval fitness function (full evaluation)
     *
     * ========================================================= */

    // the fitness function is the royal function (oneMax is a Royal Road with block of 1)
    RoyalRoadEval<Indi> fullEval(blockSize);

    /* =========================================================
     *
     * evaluation of a neighbor solution
     *
     * ========================================================= */

    // Incremental evaluation of the neighbor: fitness is modified by +1 , 0 or -1
    moRoyalRoadIncrEval<Neighbor> neighborEval(fullEval);

    /* =========================================================
     *
     * the neighborhood of a solution
     *
     * ========================================================= */

    // Exploration of the neighborhood in random order
    // at each step one bit is randomly generated
    moRndWithoutReplNeighborhood<Neighbor> neighborhood(vecSize);

    /* =========================================================
     *
     * The sampling of the search space
     *
     * ========================================================= */

    // Initial Solution of the random neutral walk
    Indi initialSol(vecSize, false);

    //          nearly 2 blocks are complete
    for (unsigned i = 0; i < blockSize - 1; i++) {
        initialSol[i] = true;
        initialSol[blockSize + i] = true;
        initialSol[2 * blockSize + i] = true;
    }
    //          first block is complete
    initialSol[blockSize - 1] = true;

    //          evaluation of the initial solution
    fullEval(initialSol);

    // Hamming distance
    eoHammingDistance<Indi> distance;

    // sampling object :
    //    - random initialization
    //    - neighborhood to compute the next step
    //    - fitness function
    //    - neighbor evaluation
    //    - number of steps of the walk
    moNeutralWalkSampling<Neighbor> sampling(initialSol, neighborhood, fullEval, neighborEval, distance, nbStep);

    /* =========================================================
     *
     * execute the sampling
     *
     * ========================================================= */

    std::cout << "Initial Solution: " << initialSol << std::endl;

    // the sampling
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
    const std::vector<Indi> & solutions = sampling.getSolutions(0);

    std::cout << "First values:" << std::endl;
    std::cout << "Solution  " << solutions[0] << std::endl;

    std::cout << "Last values:" << std::endl;
    std::cout << "Solution  " << solutions[solutions.size() - 1] << std::endl;

    // export only the solution into file
    sampling.fileExport(0, str_out + "_sol");

    // more basic statistics on the distribution:
    moStatistics statistics;

    vector< vector<double> > dist;
    vector<double> v;

    statistics.distances(solutions, distance, dist);

    for (unsigned i = 0; i < dist.size(); i++) {
        for (unsigned j = 0; j < dist.size(); j++) {
            std::cout << dist[i][j] << " " ;
            if (j < i)
                v.push_back(dist[i][j]);
        }
        std::cout << std::endl;
    }

    double min, max, avg, std;
    statistics.basic(v, min, max, avg, std);
    std::cout << "min=" << min << ", max=" << max << ", average=" << avg << ", std dev=" << std << std::endl;
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
