//-----------------------------------------------------------------------------
/** lesson1_continuator.cpp
 *
 * SV - 27/04/10 - version 1
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
#include <paradiseo/mo/neighborhood/moOrderNeighborhood.h> // visit all neighbors in increasing order of bit index

//-----------------------------------------------------------------------------
// the continuator based on the number of neighbor evaluations
#include <paradiseo/mo/eval/moEvalCounter.h>   // to import the neighbor evaluation counter
#include <paradiseo/mo/continuator/moNeighborEvalContinuator.h>

//-----------------------------------------------------------------------------
// the simple Hill-Climbing local search
#include <paradiseo/mo/algo/moSimpleHC.h>

// Declaration of types
//-----------------------------------------------------------------------------
// Indi is the typedef of the solution type like in paradisEO-eo
typedef eoBit<unsigned int> Indi;                      // bit string with unsigned fitness type
// Neighbor is the typedef of the neighbor type,
// Neighbor = How to compute the neighbor from the solution + information on it (i.e. fitness)
// all classes from paradisEO-mo use this template type
typedef moBitNeighbor<unsigned int> Neighbor ;         // bit string neighbor with unsigned fitness type


// Main function
//-----------------------------------------------------------------------------
void main_function(int argc, char **argv)
{
    /* =========================================================
     *
     * Parameters from parser
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

    // maximum number of full evaluation
    eoValueParam<unsigned int> evalParam(30, "eval", "Maximum number of neighbor evaluation", 'e');
    parser.processParam( evalParam, "Representation" );
    unsigned evalMax = evalParam.value();

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
    moOneMaxIncrEval<Neighbor> neighborEvalTmp;

    // to count the number of neighbor evaluation
    moEvalCounter<Neighbor> neighborEval(neighborEvalTmp);

    /* =========================================================
     *
     * the neighborhood of a solution
     *
     * ========================================================= */

    // Exploration of the neighborhood in increasing order of the neigbor's index:
    // bit-flip from bit 0 to bit (vecSize - 1)
    moOrderNeighborhood<Neighbor> neighborhood(vecSize);

    /* =========================================================
     *
     * the external continuators
     *
     * ========================================================= */

    moNeighborEvalContinuator<Neighbor> continuator(neighborEval, evalMax);

    /* =========================================================
     *
     * the local search algorithm
     *
     * ========================================================= */

    moSimpleHC<Neighbor> hc(neighborhood, fullEval, neighborEval, continuator);

    /* =========================================================
     *
     * executes the local search from a random solution
     *
     * ========================================================= */

    // The current solution
    Indi solution;

    // Apply random initialization
    random(solution);

    // Evaluation of the initial solution:
    // can be evaluated here, or else it will be done at the beginning of the local search
    fullEval(solution);

    // Output: the intial solution
    std::cout << "initial: " << solution << std::endl ;

    // Apply the local search on the solution !
    hc(solution);

    // Output: the final solution
    std::cout << "final:   " << solution << std::endl ;
    std::cout << "Number of neighbor evaluations during the local search: " << continuator.value() << std::endl ;

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
