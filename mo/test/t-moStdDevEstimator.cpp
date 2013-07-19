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
