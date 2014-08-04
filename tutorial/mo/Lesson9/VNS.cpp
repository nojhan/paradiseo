//-----------------------------------------------------------------------------
/** VNS.cpp
 *
 * SV - 20/08/10
 * JH - 20/08/10
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
#include <paradiseo/mo/problems/permutation/moIndexedSwapNeighbor.h>
#include <paradiseo/mo/neighborhood/moIndexNeighbor.h>
#include <paradiseo/mo/neighborhood/moRndWithoutReplNeighborhood.h>
#include <paradiseo/mo/neighborhood/moOrderNeighborhood.h>
#include <paradiseo/mo/explorer/moVNSexplorer.h>
#include <paradiseo/mo/neighborhood/moBackwardVectorVNSelection.h>
#include <paradiseo/mo/neighborhood/moForwardVectorVNSelection.h>
#include <paradiseo/mo/neighborhood/moRndVectorVNSelection.h>

//Algorithm and its components
#include <paradiseo/mo/coolingSchedule/moCoolingSchedule.h>
#include <paradiseo/mo/algo/moSimpleHC.h>
#include <paradiseo/mo/algo/moLocalSearch.h>
#include <paradiseo/mo/algo/moVNS.h>
//#include <paradiseo/mo/algo/moSimpleVNS.h>

#include <paradiseo/mo/continuator/moTimeContinuator.h>

//comparator
#include <paradiseo/mo/comparator/moSolNeighborComparator.h>

//continuators
#include <paradiseo/mo/continuator/moTrueContinuator.h>
#include <paradiseo/mo/continuator/moCheckpoint.h>
#include <paradiseo/mo/continuator/moFitnessStat.h>
#include <paradiseo/eo/utils/eoFileMonitor.h>
#include <paradiseo/mo/continuator/moCounterMonitorSaver.h>

#include <paradiseo/eo/eoSwapMutation.h>
#include <paradiseo/eo/eoShiftMutation.h>

#include <paradiseo/mo/acceptCrit/moBetterAcceptCrit.h>
#include <paradiseo/mo/acceptCrit/moAlwaysAcceptCrit.h>

//-----------------------------------------------------------------------------
// Define types of the representation solution, different neighbors and neighborhoods
//-----------------------------------------------------------------------------
typedef eoInt<eoMinimizingFitness> Queen; //Permutation (Queen's problem representation)

typedef moShiftNeighbor<Queen> shiftNeighbor; //shift Neighbor
typedef moIndexedSwapNeighbor <Queen> swapNeighbor;
typedef moIndexNeighbor<Queen> indexNeighbor;
typedef moRndWithoutReplNeighborhood<shiftNeighbor> shiftNeighborhood; //rnd shift Neighborhood (Indexed)
typedef moRndWithoutReplNeighborhood<swapNeighbor> swapNeighborhood;

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
    // you'll always get the same result, NOT a random run
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
    moFullEvalByCopy<swapNeighbor> swapEval(fullEval);

    /* =========================================================
     *
     * the neighborhood of a solution
     *
     * ========================================================= */

    shiftNeighborhood shiftNH((vecSize-1) * (vecSize-1));
    swapNeighborhood swapNH(vecSize * (vecSize-1) / 2);

    /* =========================================================
     *
     * the local search algorithm
     *
     * ========================================================= */

    moSimpleHC<shiftNeighbor> ls1(shiftNH, fullEval, shiftEval);
    moSimpleHC<swapNeighbor> ls2(swapNH, fullEval, swapEval);

    eoSwapMutation<Queen> swapMut;
    eoShiftMutation<Queen> shiftMut;

    //    moForwardVectorVNSelection<Queen> selectNH(ls1, shiftMut, true);
    //    moBackwardVectorVNSelection<Queen> selectNH(ls1, shiftMut, true);
    moRndVectorVNSelection<Queen> selectNH(ls1, shiftMut, true);

    selectNH.add(ls2, swapMut);

    moAlwaysAcceptCrit<shiftNeighbor> acceptCrit;

    //    moVNSexplorer<shiftNeighbor> explorer(selectNH, acceptCrit);

    moTimeContinuator<shiftNeighbor> cont(3);

    //    moLocalSearch<shiftNeighbor> vns(explorer, cont, fullEval);
    moVNS<shiftNeighbor> vns(selectNH, acceptCrit, fullEval, cont);

   /* moSimpleVNS<shiftNeighbor> svns(ls1, shiftMut, fullEval, cont);
    svns.add(ls2, swapMut);*/

    /* =========================================================
     *
     * execute the local search from random solution
     *
     * ========================================================= */

	Queen sol;

	init(sol);

	fullEval(sol);

	std::cout << "#########################################" << std::endl;
	std::cout << "initial sol: " << sol << std::endl ;

	vns(sol);

	std::cout << "final sol: " << sol << std::endl ;
	std::cout << "#########################################" << std::endl;

	init(sol);

	fullEval(sol);

	std::cout << "#########################################" << std::endl;
	std::cout << "initial sol: " << sol << std::endl ;

	//svns(sol);

	std::cout << "final sol: " << sol << std::endl ;
	std::cout << "#########################################" << std::endl;
//
//
//    /* =========================================================
//     *
//     * the cooling schedule of the process
//     *
//     * ========================================================= */
//
//    // initial temp, factor of decrease, number of steps without decrease, final temp.
//    moSimpleCoolingSchedule<Queen> coolingSchedule(1, 0.9, 100, 0.01);
//
//    /* =========================================================
//     *
//     * Comparator of neighbors
//     *
//     * ========================================================= */
//
//    moSolNeighborComparator<shiftNeighbor> solComparator;
//
//    /* =========================================================
//     *
//     * Example of Checkpointing
//     *
//     * ========================================================= */
//
//    moTrueContinuator<shiftNeighbor> continuator;//always continue
//    moCheckpoint<shiftNeighbor> checkpoint(continuator);
//    moFitnessStat<Queen> fitStat;
//    checkpoint.add(fitStat);
//    eoFileMonitor monitor("fitness.out", "");
//    moCounterMonitorSaver countMon(100, monitor);
//    checkpoint.add(countMon);
//    monitor.add(fitStat);
//
//    //moSA<shiftNeighbor> localSearch2(rndShiftNH, fullEval, shiftEval, coolingSchedule, solComparator, checkpoint);
//
//    init(solution2);
//
//    fullEval(solution2);
//
//    std::cout << "#########################################" << std::endl;
//    std::cout << "initial solution2: " << solution2 << std::endl ;
//
//    //localSearch2(solution2);
//
//    std::cout << "final solution2: " << solution2 << std::endl ;
//    std::cout << "#########################################" << std::endl;
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
