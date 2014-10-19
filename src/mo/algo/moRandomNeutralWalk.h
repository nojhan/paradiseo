/*
<moRandomNeutralWalk.h>
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

#ifndef _moRandomNeutralWalk_h
#define _moRandomNeutralWalk_h

#include "moLocalSearch.h"
#include "../explorer/moRandomNeutralWalkExplorer.h"
#include "../continuator/moTrueContinuator.h"
#include "../eval/moEval.h"
#include "../../eo/eoEvalFunc.h"

/**
 * Random Neutral Walk:
 * Random Neutral walk local search
 *
 * At each iteration,
 *   one random neighbor with the same fitness is selected and replace the current solution
 *   the algorithm stops when the number of steps is reached
 */
template<class Neighbor>
class moRandomNeutralWalk: public moLocalSearch<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT;
    typedef moNeighborhood<Neighbor> Neighborhood ;

    /**
     * Basic constructor for a random walk
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _eval neighbor's evaluation function
     * @param _nbStepMax number of step of the walk
     */
    moRandomNeutralWalk(Neighborhood& _neighborhood, eoEvalFunc<EOT>& _fullEval, moEval<Neighbor>& _eval, unsigned _nbStepMax):
            moLocalSearch<Neighbor>(explorer, trueCont, _fullEval),
            explorer(_neighborhood, _eval, defaultSolNeighborComp, _nbStepMax)
    {}

    /**
     * Simple constructor for a random walk
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _eval neighbor's evaluation function
     * @param _nbStepMax number of step of the walk
     * @param _cont an external continuator
     */
    moRandomNeutralWalk(Neighborhood& _neighborhood, eoEvalFunc<EOT>& _fullEval, moEval<Neighbor>& _eval, unsigned _nbStepMax, moContinuator<Neighbor>& _cont):
            moLocalSearch<Neighbor>(explorer, _cont, _fullEval),
            explorer(_neighborhood, _eval, defaultSolNeighborComp, _nbStepMax)
    {}

    /**
     * General constructor for a random walk
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _eval neighbor's evaluation function
     * @param _nbStepMax number of step of the walk
     * @param _cont an external continuator
     * @param _comp a solution vs neighbor comparator
     */
    moRandomNeutralWalk(Neighborhood& _neighborhood, eoEvalFunc<EOT>& _fullEval, moEval<Neighbor>& _eval, unsigned _nbStepMax, moContinuator<Neighbor>& _cont, moSolNeighborComparator<Neighbor>& _comp):
            moLocalSearch<Neighbor>(explorer, _cont, _fullEval),
            explorer(_neighborhood, _eval, _comp, _nbStepMax)
    {}

private:
    // always true continuator
    moTrueContinuator<Neighbor> trueCont;
    // the explorer of the random walk
    moRandomNeutralWalkExplorer<Neighbor> explorer;
    // compare the fitness values of the solution and the neighbor
    moSolNeighborComparator<Neighbor> defaultSolNeighborComp;
};

#endif
