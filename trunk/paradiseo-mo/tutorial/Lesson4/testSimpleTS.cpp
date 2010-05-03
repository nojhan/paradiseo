//-----------------------------------------------------------------------------
/** testSimpleHC.cpp
 *
 * SV - 12/01/10
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
//Representation and initializer
#include <eoInt.h>
#include <eoInit.h>
#include <eoScalarFitness.h>

// fitness function
#include <eval/queenEval.h>
#include <eval/moFullEvalByModif.h>
#include <eval/moFullEvalByCopy.h>

//Neighbors and Neighborhoods
#include <problems/permutation/moShiftNeighbor.h>
#include <neighborhood/moRndWithReplNeighborhood.h>
#include <neighborhood/moOrderNeighborhood.h>

//Algorithm and its components
#include <coolingSchedule/moCoolingSchedule.h>
//#include <algo/moTS.h>

//comparator
#include <comparator/moSolNeighborComparator.h>
#include <comparator/moNeighborComparator.h>

//continuators
#include <continuator/moTrueContinuator.h>
#include <continuator/moCheckpoint.h>
#include <continuator/moFitnessStat.h>
#include <utils/eoFileMonitor.h>
#include <continuator/moCounterMonitorSaver.h>

//mo eval
#include <eval/moFullEvalByModif.h>
#include <eval/moFullEvalByCopy.h>

#include <mo>

// REPRESENTATION
//-----------------------------------------------------------------------------
typedef eoInt<eoMinimizingFitness> Queen; //Permutation (Queen's problem representation)

typedef moShiftNeighbor<Queen> shiftNeighbor; //shift Neighbor
typedef moOrderNeighborhood<shiftNeighbor> orderShiftNeighborhood; //rnd shift Neighborhood (Indexed)

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

    // size tabu list
    eoValueParam<unsigned int> sizeTabuListParam(7, "sizeTabuList", "size of the tabu list", 'T');
    parser.processParam( sizeTabuListParam, "Search Parameters" );
    unsigned sizeTabuList = sizeTabuListParam.value();

    // Time Limit
    eoValueParam<unsigned int> timeLimitParam(1, "timeLimit", "time limits", 'T');
    parser.processParam( timeLimitParam, "Search Parameters" );
    unsigned timeLimit = timeLimitParam.value();

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

    queenEval<Queen> fullEval;


    /* =========================================================
     *
     * Initilisation of the solution
     *
     * ========================================================= */

    eoInitPermutation<Queen> init(vecSize);


    /* =========================================================
     *
     * evaluation of a neighbor solution
     *
     * ========================================================= */

    moFullEvalByCopy<shiftNeighbor> shiftEval(fullEval);

    //An eval by copy can be used instead of the eval by modif
    //moFullEvalByCopy<Neighbor> fulleval(eval);


    /* =========================================================
     *
     * the neighborhood of a solution
     *
     * ========================================================= */

    orderShiftNeighborhood rndShiftNH(pow(vecSize-1, 2));

    /* =========================================================
     *
     * Comparator of neighbors
     *
     * ========================================================= */

    moSolNeighborComparator<shiftNeighbor> solComparator;
    moNeighborComparator<shiftNeighbor> comparator;

    /* =========================================================
     *
     * a neighborhood explorer solution
     *
     * ========================================================= */

    moNeighborVectorTabuList<shiftNeighbor> tl(sizeTabuList,0);
    moDummyIntensification<shiftNeighbor> inten;
    moDummyDiversification<shiftNeighbor> div;
    moBestImprAspiration<shiftNeighbor> asp;
    //moTSexplorer<shiftNeighbor> explorer(rndShiftNH, shiftEval, comparator, solComparator, tl, inten, div, asp);


    /* =========================================================
     *
     * the local search algorithm
     *
     * ========================================================= */

    moTimeContinuator<shiftNeighbor> continuator(timeLimit);

    moTS<shiftNeighbor> localSearch(rndShiftNH, fullEval, shiftEval, comparator, solComparator, continuator, tl, inten, div, asp);

    /* =========================================================
     *
     * execute the local search from random sollution
     *
     * ========================================================= */

    Queen solution;

    init(solution);

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
