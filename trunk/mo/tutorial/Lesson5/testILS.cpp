//-----------------------------------------------------------------------------
/** testILS.cpp
 *
 * SV - 12/01/10
 * JH - 04/05/10
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
#include <ga/eoBitOp.h>

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
#include <neighborhood/moOrderNeighborhood.h>

//Mutation
#include <eoSwapMutation.h>

//Algorithm and its components
#include <algo/moTS.h>
#include <algo/moILS.h>

//mo eval
#include <eval/moFullEvalByCopy.h>

#include <perturb/moMonOpPerturb.h>
#include <perturb/moRestartPerturb.h>
#include <perturb/moNeighborhoodPerturb.h>
#include <acceptCrit/moAlwaysAcceptCrit.h>
#include <acceptCrit/moBetterAcceptCrit.h>

#include <continuator/moIterContinuator.h>

// REPRESENTATION
//-----------------------------------------------------------------------------
typedef eoInt<eoMinimizingFitness> Queen; //Permutation (Queen's problem representation)

typedef moShiftNeighbor<Queen> shiftNeighbor; //shift Neighbor
typedef moOrderNeighborhood<shiftNeighbor> orderShiftNeighborhood; //order shift Neighborhood (Indexed)

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
     * Full evaluation fitness function
     *
     * ========================================================= */

    queenEval<Queen> fullEval;


    /* =========================================================
     *
     * Initializer of a solution
     *
     * ========================================================= */

    eoInitPermutation<Queen> init(vecSize);


    /* =========================================================
     *
     * Declare and init solutions
     *
     * ========================================================= */

    Queen sol1;
    Queen sol2;
    Queen sol3;

    //random initialization
    init(sol1);
    init(sol2);
    init(sol3);

    //evaluation
    fullEval(sol1);
    fullEval(sol2);
    fullEval(sol3);

    /* =========================================================
     *
     * evaluation of a neighbor solution
     *
     * ========================================================= */

    moFullEvalByCopy<shiftNeighbor> shiftEval(fullEval);

    /* =========================================================
     *
     * the neighborhood of a solution
     *
     * ========================================================= */

    orderShiftNeighborhood orderShiftNH((vecSize-1) * (vecSize-1));


    /* =========================================================
     *
     * the local search algorithms
     *
     * ========================================================= */

    //Basic Constructor of the Tabu Search
    moTS<shiftNeighbor> ts(orderShiftNH, fullEval, shiftEval, 1, 7);

    eoSwapMutation<Queen> mut;

    //Basic Constructor of the Iterated Local Search
    moILS<shiftNeighbor> localSearch1(ts, fullEval, mut, 3);


    //Simple Constructor of the Iterated Local Search
    //Be carefull, template of the continuator must be a dummyNeighbor!!!
    moIterContinuator<moDummyNeighbor<Queen> > cont(4, false);
    moILS<shiftNeighbor> localSearch2(ts, fullEval, mut, cont);

    //General Constructor of the Iterated Local Search
    moMonOpPerturb<shiftNeighbor> perturb(mut, fullEval);

    moSolComparator<Queen> solComp;
    moBetterAcceptCrit<shiftNeighbor> accept(solComp);

    moILS<shiftNeighbor> localSearch3(ts, fullEval, cont, perturb, accept);

    std::cout << "Iterated Local Search 1:" << std::endl;
    std::cout << "--------------" << std::endl;
    std::cout << "initial: " << sol1 << std::endl ;
    localSearch1(sol1);
    std::cout << "final:   " << sol1 << std::endl << std::endl;

    std::cout << "Iterated Local Search 2:" << std::endl;
    std::cout << "--------------" << std::endl;
    std::cout << "initial: " << sol2 << std::endl ;
    localSearch2(sol2);
    std::cout << "final:   " << sol2 << std::endl << std::endl;

    std::cout << "Iterated Local Search 3:" << std::endl;
    std::cout << "--------------" << std::endl;
    std::cout << "initial: " << sol3 << std::endl ;
    localSearch3(sol3);
    std::cout << "final:   " << sol3 << std::endl << std::endl;

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
