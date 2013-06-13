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

double objective_function(const EOT & sol)
{
    double sum = 0;

    for ( size_t i = 0; i < sol.size(); ++i )
	{
		//std::cout << sol[i] << std::endl;
	    sum += sol[i] * sol[i];
	}

    return sum;
}

int main(int ac, char** av)
{
    //moNeighborhoodStat<Neighbor> nhStat
    moFitnessVarianceStat<EOT> stat;
    eoEvalFuncPtr< EOT, double > eval( objective_function );
    EOT solution(2, 5);
    eval(solution);
    stat(solution);
    solution[0] = solution[1] = 0;
    solution.invalidate();
    eval(solution);
    stat(solution);
    //assert(stat.value() == 1);
    std::cout << "var: " << stat.value() << std::endl;
    assert(stat.value() == 625);
}
