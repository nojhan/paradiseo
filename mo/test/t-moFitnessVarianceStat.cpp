/*

(c) Thales group, 2010

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation;
    version 2 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Contact: http://eodev.sourceforge.net

Authors:
Lionel Parreaux <lionel.parreaux@gmail.com>

*/

//-----------------------------------------------------------------------------
// t-moFitnessVarianceStat.cpp
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
	    sum += sol[i] * sol[i];
	}

    return sum;
}

int main(int ac, char** av)
{
    moFitnessVarianceStat<EOT> stat;
    eoEvalFuncPtr< EOT, double > eval( objective_function );
    EOT solution(2, 5);
    eval(solution);
    stat(solution);
    solution[0] = solution[1] = 0;
    solution.invalidate();
    eval(solution);
    stat(solution);
    std::cout << "var: " << stat.value() << std::endl;
    assert(stat.value() == 625);
    std::cout << "[t-moFitnessNeighborStat] => OK" << std::endl;
}
