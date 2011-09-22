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
#include <problems/permutation/moIndexedSwapNeighbor.h>
#include <neighborhood/moIndexNeighbor.h>
#include <neighborhood/moRndWithoutReplNeighborhood.h>
#include <neighborhood/moOrderNeighborhood.h>
#include <explorer/moVNSexplorer.h>
#include <neighborhood/moBackwardVectorVNSelection.h>
#include <neighborhood/moForwardVectorVNSelection.h>
#include <neighborhood/moRndVectorVNSelection.h>

//Algorithm and its components
#include <coolingSchedule/moCoolingSchedule.h>
#include <algo/moSimpleHC.h>
#include <algo/moLocalSearch.h>
#include <algo/moVNS.h>
//#include <algo/moSimpleVNS.h>

#include <continuator/moTimeContinuator.h>

//comparator
#include <comparator/moSolNeighborComparator.h>

//continuators
#include <continuator/moTrueContinuator.h>
#include <continuator/moCheckpoint.h>
#include <continuator/moFitnessStat.h>
#include <utils/eoFileMonitor.h>
#include <continuator/moCounterMonitorSaver.h>

#include <eoSwapMutation.h>
#include <eoShiftMutation.h>

#include <acceptCrit/moBetterAcceptCrit.h>
#include <acceptCrit/moAlwaysAcceptCrit.h>

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
