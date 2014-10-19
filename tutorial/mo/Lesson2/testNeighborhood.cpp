//-----------------------------------------------------------------------------
/** testNeighborhood.cpp
 *
 * JH - 09/04/10
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
#include <paradiseo/eo.h>
#include <paradiseo/eo/ga.h>

using namespace std;

//----------------------------------------------------------------------------

//Representation and initializer
#include <paradiseo/eo/eoInt.h>
#include <paradiseo/eo/eoInit.h>

// fitness function
#include <paradiseo/problems/eval/queenEval.h>
#include <paradiseo/mo/eval/moFullEvalByModif.h>
#include <paradiseo/mo/eval/moFullEvalByCopy.h>

//Neighbors and Neighborhoods
#include <paradiseo/mo/problems/permutation/moShiftNeighbor.h>
#include <paradiseo/mo/problems/permutation/moSwapNeighbor.h>
#include <paradiseo/mo/problems/permutation/moSwapNeighborhood.h>
#include <paradiseo/mo/neighborhood/moRndWithReplNeighborhood.h>
#include <paradiseo/mo/neighborhood/moRndWithoutReplNeighborhood.h>
#include <paradiseo/mo/neighborhood/moOrderNeighborhood.h>


// Define types of the representation solution, different neighbors and neighborhoods
//-----------------------------------------------------------------------------
typedef eoInt<unsigned int> Queen; //Permutation (Queen's problem representation)

typedef moSwapNeighbor<Queen> swapNeighbor ; //swap Neighbor
typedef moSwapNeighborhood<Queen> swapNeighborhood; //classical swap Neighborhood

typedef moShiftNeighbor<Queen> shiftNeighbor; //shift Neighbor
typedef moOrderNeighborhood<shiftNeighbor> orderShiftNeighborhood; //order shift Neighborhood (Indexed)
typedef moRndWithoutReplNeighborhood<shiftNeighbor> rndWithoutReplShiftNeighborhood; //random without replacement shift Neighborhood (Indexed)
typedef moRndWithReplNeighborhood<shiftNeighbor> rndWithReplShiftNeighborhood; //random with replacement shift Neighborhood (Indexed)

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
     * Eval fitness function
     *
     * ========================================================= */

    queenEval<Queen> fullEval;


    /* =========================================================
     *
     * Initializer of the solution
     *
     * ========================================================= */

    eoInitPermutation<Queen> init(vecSize);


    /* =========================================================
     *
     * evaluation operators of a neighbor solution
     *
     * ========================================================= */

    moFullEvalByModif<swapNeighbor> swapEval(fullEval);
    moFullEvalByCopy<shiftNeighbor> shiftEval(fullEval);


    /* =========================================================
     *
     * Neighbors and Neighborhoods
     *
     * ========================================================= */

    swapNeighborhood swapNH;
    orderShiftNeighborhood orderShiftNH((vecSize-1) * (vecSize-1));
    rndWithoutReplShiftNeighborhood rndNoReplShiftNH((vecSize-1) * (vecSize-1));
    rndWithReplShiftNeighborhood rndReplShiftNH((vecSize-1) * (vecSize-1));


    /* =========================================================
     *
     * Init and eval a Queen
     *
     * ========================================================= */

    Queen solution;

    init(solution);

    fullEval(solution);

    std::cout << "Initial Solution:" << std::endl;
    std::cout << solution << std::endl << std::endl;

    /* =========================================================
     *
     * Use classical Neighbor and Neighborhood (swap)
     *
     * ========================================================= */

    std::cout << "SWAP NEIGHBORHOOD" << std::endl;
    std::cout << "-----------------" << std::endl;
    std::cout << "Neighbors List: (Neighbor -> fitness)" << std::endl;

    swapNeighbor n1;
    swapNH.init(solution, n1);
    swapEval(solution,n1);
    n1.print();
    while (swapNH.cont(solution)) {
        swapNH.next(solution, n1);
        swapEval(solution,n1);
        n1.print();
    }

    /* =========================================================
     *
     * Use indexed Neighborhood with shift operator
     *
     * ========================================================= */

    std::cout << "\nSHIFT ORDER NEIGHBORHOOD" << std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout << "Neighbors List: (key: Neighbor -> fitness)" << std::endl;

    shiftNeighbor n2;

    orderShiftNH.init(solution, n2);
    shiftEval(solution,n2);
    n2.print();
    while (orderShiftNH.cont(solution)) {
        orderShiftNH.next(solution, n2);
        shiftEval(solution,n2);
        n2.print();
    }

    std::cout << "\nSHIFT RANDOM WITHOUT REPLACEMENT NEIGHBORHOOD" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "Neighbors List: (key: Neighbor -> fitness)" << std::endl;

    rndNoReplShiftNH.init(solution, n2);
    shiftEval(solution,n2);
    n2.print();
    while (rndNoReplShiftNH.cont(solution)) {
        rndNoReplShiftNH.next(solution, n2);
        shiftEval(solution,n2);
        n2.print();
    }

    std::cout << "\nSHIFT RANDOM WITH REPLACEMENT NEIGHBORHOOD" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "Neighbors List: (key: Neighbor -> fitness)" << std::endl;

    rndReplShiftNH.init(solution, n2);
    shiftEval(solution,n2);
    n2.print();
    for (unsigned int i=0; i<100; i++) {
        rndReplShiftNH.next(solution, n2);
        shiftEval(solution,n2);
        n2.print();
    }

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
