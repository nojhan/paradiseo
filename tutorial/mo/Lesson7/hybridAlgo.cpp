//-----------------------------------------------------------------------------
/** testILS.cpp
 *
 * SV - 12/01/10
 * JH - 06/05/10
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
#include <paradiseo/eo/ga/eoBitOp.h>

using namespace std;

//-----------------------------------------------------------------------------
//Representation and initializer
#include <paradiseo/eo/eoInt.h>
#include <paradiseo/eo/eoInit.h>
#include <paradiseo/eo/eoScalarFitness.h>

// fitness function
#include <paradiseo/problems/eval/queenEval.h>
#include <paradiseo/mo/eval/moFullEvalByModif.h>
#include <paradiseo/mo/eval/moFullEvalByCopy.h>

//Neighbors and Neighborhoods
#include <paradiseo/mo/problems/permutation/moShiftNeighbor.h>
#include <paradiseo/mo/neighborhood/moOrderNeighborhood.h>

//Mutation
#include <paradiseo/eo/eoSwapMutation.h>
#include <paradiseo/eo/eoOrderXover.h>

//Algorithm and its components
#include <paradiseo/mo/algo/moFirstImprHC.h>

//mo eval
#include <paradiseo/mo/eval/moFullEvalByCopy.h>

#include <paradiseo/mo/continuator/moTrueContinuator.h>

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
     * Declare and init a population
     *
     * ========================================================= */

    eoPop<Queen> pop;

    Queen tmp;

    for (unsigned int i=0; i<20; i++) {
        init(tmp);
        fullEval(tmp);
        pop.push_back(tmp);
    }

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
     * the local search algorithm
     *
     * ========================================================= */

    //Basic Constructor a first improvement hill climber
    moFirstImprHC<shiftNeighbor> hc(orderShiftNH, fullEval, shiftEval);

    /* =========================================================
     *
     * the evolutionary algorithm
     *
     * ========================================================= */

    //continuator
    eoGenContinue<Queen> EAcont(50);

    //selection
    eoDetTournamentSelect<Queen> selectOne(2);
    eoSelectMany<Queen> select(selectOne, 1);

    //crossover
    eoOrderXover<Queen> cross;

    //transform operator (the hill climber replace the mutation operator)
    eoSGATransform<Queen> transform(cross, 0.3, hc, 0.7);

    //replacement
    eoGenerationalReplacement<Queen> repl;

    //easyEA
    eoEasyEA<Queen> hybridAlgo(EAcont, fullEval, select, transform, repl);


    std::cout << "INITIAL POPULATION:" << std::endl;
    std::cout << "-------------------" << std::endl;

    for (unsigned int i=0; i<pop.size(); i++)
        std::cout << pop[i] << std::endl;

    hybridAlgo(pop);

    std::cout << std::endl;
    std::cout << "FINAL POPULATION:" << std::endl;
    std::cout << "-------------------" << std::endl;
    for (unsigned int i=0; i<pop.size(); i++)
        std::cout << pop[i] << std::endl;


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
