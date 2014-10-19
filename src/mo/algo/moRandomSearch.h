/*
<moRandomSearch.h>
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

#ifndef _moRandomSearch_h
#define _moRandomSearch_h

#include "moLocalSearch.h"
#include "../explorer/moRandomSearchExplorer.h"
#include "../continuator/moTrueContinuator.h"
#include "../../eo/eoInit.h"
#include "../../eo/eoEvalFunc.h"

/**
 * Random Search:
 * Pure random search local search
 *
 * At each iteration,
 *   one random solution is selected and replace the current solution
 *   the algorithm stops when the number of solution is reached
 */
template<class Neighbor>
class moRandomSearch: public moLocalSearch<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT;

    /**
     * Simple constructor for a random search
     * @param _init the solution initializer, to explore at random the search space
     * @param _fullEval the full evaluation function
     * @param _nbSolMax number of solutions
     */
    moRandomSearch(eoInit<EOT> & _init, eoEvalFunc<EOT>& _fullEval, unsigned _nbSolMax):
            moLocalSearch<Neighbor>(explorer, trueCont, _fullEval),
            explorer(_init, _fullEval, _nbSolMax>0?_nbSolMax - 1:0)
    {}

    /**
     * General constructor for a random search
     * @param _init the solution initializer, to explore at random the search space
     * @param _fullEval the full evaluation function
     * @param _nbSolMax number of solutions
     * @param _cont external continuator
     */
    moRandomSearch(eoInit<EOT> & _init, eoEvalFunc<EOT>& _fullEval, unsigned _nbSolMax, moContinuator<Neighbor>& _cont):
            moLocalSearch<Neighbor>(explorer, _cont, _fullEval),
            explorer(_init, _fullEval, _nbSolMax>0?_nbSolMax - 1:0)
    {}

    /**
     * @return name of the class
     */
    virtual std::string className(void) const {
        return "moRandomSearch";
    }

private:
    // always true continuator
    moTrueContinuator<Neighbor> trueCont;
    // the explorer of the random walk
    moRandomSearchExplorer<Neighbor> explorer;
};

#endif
