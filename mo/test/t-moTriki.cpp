//============================================================================
// Name        : Trikitest.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

//#define HAVE_GNUPLOT

#include <iostream>
using namespace std;

#include <mo>
#include <eo>

//Representation and initializer
#include <eoInt.h>
#include <eoInit.h>
#include <eoScalarFitness.h>

/*
// fitness function
#include <eval/queenEval.h>
#include <eval/moFullEvalByModif.h>
#include <eval/moFullEvalByCopy.h>

//Neighbors and Neighborhoods
#include <problems/permutation/moShiftNeighbor.h>
#include <neighborhood/moRndWithReplNeighborhood.h>

//Algorithm and its components
#include <coolingSchedule/moCoolingSchedule.h>
#include <algo/moSA.h>

//comparator
#include <comparator/moSolNeighborComparator.h>

//continuators
#include <continuator/moTrueContinuator.h>
#include <continuator/moCheckpoint.h>
#include <continuator/moFitnessStat.h>
#include <utils/eoFileMonitor.h>
#include <continuator/moCounterMonitorSaver.h>
*/

//-----------------------------------------------------------------------------
// Define types of the representation solution, different neighbors and neighborhoods
//-----------------------------------------------------------------------------
typedef eoInt<eoMinimizingFitness> Queen; //Permutation (Queen's problem representation)

typedef moShiftNeighbor<Queen> shiftNeighbor; //shift Neighbor
typedef moRndWithReplNeighborhood<shiftNeighbor> rndShiftNeighborhood; //rnd shift Neighborhood (Indexed)




int main() {
    //cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
    //return 0;
    
    unsigned vecSize = 8;
    
    
    queenEval<Queen> fullEval;
    
    
    /* =========================================================
     *
     * Initilization of the solution
     *
     * ========================================================= */
    
    eoInitPermutation<Queen> init(vecSize);
    
    /* =========================================================
     *
     * evaluation of a neighbor solution
     *
     * ========================================================= */
    
    moFullEvalByCopy<shiftNeighbor> shiftEval(fullEval); /// by default
    
    /* =========================================================
     *
     * the neighborhood of a solution
     *
     * ========================================================= */
    
    rndShiftNeighborhood rndShiftNH((vecSize-1) * (vecSize-1));
    
    /* =========================================================
     *
     * the cooling schedule of the process
     *
     * ========================================================= */

    /* =========================================================
     *
     * the local search algorithm
     *
     * ========================================================= */
    
    /* =========================================================
     *
     * execute the local search from random solution
     *
     * ========================================================= */
    
    Queen solution1;
    
    init(solution1);
    
    fullEval(solution1);
    
    moStdDevEstimator<Queen, shiftNeighbor> initTemp (500, rndShiftNH, fullEval);
    
    moTrueContinuator<shiftNeighbor> continuator;
    moCheckpoint<shiftNeighbor> checkpoint(continuator);
    moFitnessStat<Queen> fitStat;
    checkpoint.add(fitStat);
    eoFileMonitor monitor("triki.out", "");
    eoGnuplot1DMonitor monitor2("trikignu.out", true);
    moCounterMonitorSaver countMon(100, monitor);
    checkpoint.add(countMon);
    moCounterMonitorSaver gnuMon (1, monitor2);
    checkpoint.add(gnuMon);
    monitor.add(fitStat);
    monitor2.add(fitStat);
    //#ifdef HAVE_GNUPLOT
    
    
    moTrikiCoolingSchedule<Queen, shiftNeighbor> coolingSchedule(rndShiftNH, shiftEval, initTemp(solution1));
    moSA<shiftNeighbor> localSearch1(rndShiftNH, fullEval, coolingSchedule, shiftEval, checkpoint);
    
    
    std::cout << "#########################################" << std::endl;
    std::cout << "initial solution1: " << solution1 << std::endl ;
    
    localSearch1(solution1);
    
    std::cout << "final solution1: " << solution1 << std::endl ;
    std::cout << "#########################################" << std::endl;
    
    

}








