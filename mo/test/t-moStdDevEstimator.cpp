//-----------------------------------------------------------------------------
// t-moStdDevEstimator.cpp
//-----------------------------------------------------------------------------

#include <eo>
#include "es/eoReal.h"
#include "continuator/moFitnessVarianceStat.h"
#include "neighborhood/moRealNeighbor.h"
#include "neighborhood/moRealNeighborhood.h"

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

//Algorithm and its components
#include <coolingSchedule/moCoolingSchedule.h>
#include <algo/moSA.h>

//comparator
#include <comparator/moSolNeighborComparator.h>


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
    
    //moFullEvalByCopy<shiftNeighbor> shiftEval(fullEval); /// by default
    
    rndShiftNeighborhood rndShiftNH((vecSize-1) * (vecSize-1));
    
    Queen solution;
    
    init(solution);
    
    fullEval(solution);

    moStdDevEstimator<Queen, shiftNeighbor> initTemp (500, rndShiftNH, fullEval);
    
    
    std::cout << "temp: " << initTemp(solution) << std::endl;
    //assert(stat.value() == 625);
}
