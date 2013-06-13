//-----------------------------------------------------------------------------
// t-moStdDevEstimator.cpp
//-----------------------------------------------------------------------------

#include <eo>
#include "es/eoReal.h"
#include "neighborhood/moRealNeighbor.h"

//Representation and initializer
#include <eoInt.h>
//#include <eoInit.h>
#include <eoScalarFitness.h>

// fitness function
#include <eval/queenEval.h>

//Neighbors and Neighborhoods
#include <problems/permutation/moShiftNeighbor.h>
#include <neighborhood/moRndWithReplNeighborhood.h>

//Sampling
#include <sampling/moStdDevEstimator.h>


//-----------------------------------------------------------------------------
// Define types of the representation solution, different neighbors and neighborhoods
//-----------------------------------------------------------------------------
typedef eoInt<eoMinimizingFitness> Queen; //Permutation (Queen's problem representation)

typedef moShiftNeighbor<Queen> shiftNeighbor; //shift Neighbor
typedef moRndWithReplNeighborhood<shiftNeighbor> rndShiftNeighborhood; //rnd shift Neighborhood (Indexed)

//-----------------------------------------------------------------------------

typedef eoReal< eoMinimizingFitness > EOT;
typedef moRealNeighbor< EOT > Neighbor;

int main(int ac, char** av)
{
    unsigned vecSize = 8;
    
    queenEval<Queen> fullEval;
    
    eoInitPermutation<Queen> init(vecSize);
    
    rndShiftNeighborhood rndShiftNH((vecSize-1) * (vecSize-1));
    
    Queen solution;
    
    init(solution);
    
    fullEval(solution);

    moStdDevEstimator<Queen, shiftNeighbor> initTemp (500, rndShiftNH, fullEval);
    
    double temp = initTemp(solution);
    
    std::cout << "temp: " << temp << std::endl;

    assert(temp >= 0);
}
