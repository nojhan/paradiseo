//-----------------------------------------------------------------------------
// t-moFitnessNeighborStat.cpp
//-----------------------------------------------------------------------------

#include <eo>
#include "es/eoReal.h"
#include "continuator/moFitnessVarianceStat.h"
#include "neighborhood/moRealNeighbor.h"
#include "neighborhood/moRealNeighborhood.h"

//-----------------------------------------------------------------------------

typedef eoReal< eoMinimizingFitness > EOT;
typedef moRealNeighbor< EOT > Neighbor;

int main(int ac, char** av)
{
    //moNeighborhoodStat<Neighbor> nhStat
    moFitnessVarianceStat<EOT> stat;
    EOT solution(2, 5);
    stat(solution);
    //assert(stat.value() == 1);
    std::cout << "ok " << stat.value() << std::endl;
}
