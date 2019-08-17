/*
<moMetropolisHasting.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sebastien Verel, Arnaud Liefooghe, Jeremie Humeau

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  ue,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.
The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

ParadisEO WebSite : http://paradiseo.gforge.inria.fr
Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef _moMetropolisHasting_h
#define _moMetropolisHasting_h

#include "moLocalSearch.h"
#include "../explorer/moMetropolisHastingExplorer.h"
#include "../continuator/moTrueContinuator.h"
#include "../eval/moEval.h"
#include "../../eo/eoEvalFunc.h"

/**
 * Metropolis-Hasting local search
 * Only the symetric case is considered when Q(x,y) = Q(y,x)
 * Fitness must be > 0
 *
 * At each iteration,
 *   one of the random solution in the neighborhood is selected
 *   if the selected neighbor have higher or equal fitness than the current solution
 *       then the solution is replaced by the selected neighbor
 *   if a random number from [0,1] is lower than fitness(neighbor) / fitness(solution)
 *       then the solution is replaced by the selected neighbor
 *   the algorithm stops when the number of iterations is too large
 */
template<class Neighbor>
class moMetropolisHasting: public moLocalSearch<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT;
    typedef moNeighborhood<Neighbor> Neighborhood ;

    /**
     * Basic constructor of the Metropolis-Hasting
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _eval neighbor's evaluation function
     * @param _nbStep maximum step to do
     */
    moMetropolisHasting(Neighborhood& _neighborhood, eoEvalFunc<EOT>& _fullEval, moEval<Neighbor>& _eval, unsigned int _nbStep):
            moLocalSearch<Neighbor>(explorer, trueCont, _fullEval),
            explorer(_neighborhood, _eval, defaultNeighborComp, defaultSolNeighborComp, _nbStep)
    {}

    /**
     * Simple constructor of the Metropolis-Hasting
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _eval neighbor's evaluation function
     * @param _nbStep maximum step to do
     * @param _cont an external continuator
     */
    moMetropolisHasting(Neighborhood& _neighborhood, eoEvalFunc<EOT>& _fullEval, moEval<Neighbor>& _eval, unsigned int _nbStep, moContinuator<Neighbor>& _cont):
            moLocalSearch<Neighbor>(explorer, _cont, _fullEval),
            explorer(_neighborhood, _eval, defaultNeighborComp, defaultSolNeighborComp, _nbStep)
    {}

    /**
     * General constructor of the Metropolis-Hasting
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _eval neighbor's evaluation function
     * @param _nbStep maximum step to do
     * @param _cont an external continuator
     * @param _compN  a neighbor vs neighbor comparator
     * @param _compSN a solution vs neighbor comparator
     */
    moMetropolisHasting(Neighborhood& _neighborhood, eoEvalFunc<EOT>& _fullEval, moEval<Neighbor>& _eval, unsigned int _nbStep, moContinuator<Neighbor>& _cont, moNeighborComparator<Neighbor>& _compN, moSolNeighborComparator<Neighbor>& _compSN):
            moLocalSearch<Neighbor>(explorer, _cont, _fullEval),
            explorer(_neighborhood, _eval, _compN, _compSN, _nbStep)
    {}

private:
    // always true continuator
    moTrueContinuator<Neighbor> trueCont;
    // compare the fitness values of neighbors
    moNeighborComparator<Neighbor> defaultNeighborComp;
    // compare the fitness values of the solution and the neighbor
    moSolNeighborComparator<Neighbor> defaultSolNeighborComp;
    // MetropolisHasting explorer
    moMetropolisHastingExplorer<Neighbor> explorer;
};

#endif
