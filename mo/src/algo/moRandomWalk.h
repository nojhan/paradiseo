/*
<moRandomWalk.h>
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

#ifndef _moRandomWalk_h
#define _moRandomWalk_h

#include "moLocalSearch.h"
#include "../explorer/moRandomWalkExplorer.h"
#include "../continuator/moIterContinuator.h"
#include "../eval/moEval.h"
#include <paradiseo/eo/eoEvalFunc.h>

/**
 * Random Walk:
 * Random walk local search
 *
 * At each iteration,
 *   one random neighbor is selected and replace the current solution
 *   the algorithm stops when the number of steps is reached
 */
template<class Neighbor>
class moRandomWalk: public moLocalSearch<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT;
    typedef moNeighborhood<Neighbor> Neighborhood ;

    /**
     * Simple constructor for a random walk
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _eval neighbor's evaluation function
     * @param _nbStepMax number of step of the walk
     */
    moRandomWalk(Neighborhood& _neighborhood, eoEvalFunc<EOT>& _fullEval, moEval<Neighbor>& _eval, unsigned _nbStepMax):
      moLocalSearch<Neighbor>(explorer, iterCont, _fullEval),
      iterCont(_nbStepMax, false),
      explorer(_neighborhood, _eval)
    {}

    /**
     * General constructor for a random walk
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _eval neighbor's evaluation function
     * @param _cont a user-defined continuator
     */
    moRandomWalk(Neighborhood& _neighborhood, eoEvalFunc<EOT>& _fullEval, moEval<Neighbor>& _eval, moContinuator<Neighbor>& _cont):
            moLocalSearch<Neighbor>(explorer, _cont, _fullEval),
	    iterCont(0),
            explorer(_neighborhood, _eval)
    {}

private:
    // the continuator to stop on a maximum number of step
    moIterContinuator<Neighbor> iterCont;
    // the explorer of the random walk
    moRandomWalkExplorer<Neighbor> explorer;
};

#endif
