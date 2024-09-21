/*
<moSA.h>
Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

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

#ifndef _moSA_h
#define _moSA_h

#include <algo/moLocalSearch.h>
#include <explorer/moSAexplorer.h>
#include <coolingSchedule/moCoolingSchedule.h>
#include <coolingSchedule/moSimpleCoolingSchedule.h>
#include <continuator/moTrueContinuator.h>
#include <eval/moEval.h>
#include <eoEvalFunc.h>

/**
 * Simulated Annealing
 */
template<class Neighbor>
class moSA: public moLocalSearch<Neighbor>
{
public:

    typedef typename Neighbor::EOT EOT;
    typedef moNeighborhood<Neighbor> Neighborhood ;


    /**
     * Basic constructor for a simulated annealing
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _eval neighbor's evaluation function
     * @param _initT initial temperature for cooling schedule (default = 10)
     * @param _alpha factor of decreasing for cooling schedule (default = 0.9)
     * @param _span number of iteration with equal temperature for cooling schedule (default = 100)
     * @param _finalT final temperature, threshold of the stopping criteria for cooling schedule (default = 0.01)
     */
    moSA(Neighborhood& _neighborhood, eoEvalFunc<EOT>& _fullEval, moEval<Neighbor>& _eval, double _initT=10, double _alpha=0.9, unsigned _span=100, double _finalT=0.01):
            moLocalSearch<Neighbor>(explorer, trueCont, _fullEval),
            defaultCool(_initT, _alpha, _span, _finalT),
            explorer(_neighborhood, _eval, defaultSolNeighborComp, defaultCool)
    {}

    /**
     * Simple constructor for a simulated annealing
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _eval neighbor's evaluation function
     * @param _cool a cooling schedule
     */
    moSA(Neighborhood& _neighborhood, eoEvalFunc<EOT>& _fullEval, moEval<Neighbor>& _eval, moCoolingSchedule<EOT>& _cool):
            moLocalSearch<Neighbor>(explorer, trueCont, _fullEval),
            defaultCool(0, 0, 0, 0),
            explorer(_neighborhood, _eval, defaultSolNeighborComp, _cool)
    {}

    /**
     * Constructor without cooling schedule, but with a continuator.
     *
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _eval neighbor's evaluation function
     * @param _cont an external continuator
     */
    moSA(Neighborhood& _neighborhood, eoEvalFunc<EOT>& _fullEval, moEval<Neighbor>& _eval, moContinuator<Neighbor>& _cont):
            moLocalSearch<Neighbor>(explorer, _cont, _fullEval),
            defaultCool(0, 0, 0, 0),
            explorer(_neighborhood, _eval, defaultSolNeighborComp, defaultCool)
    {}

    /**
     * General constructor for a simulated annealing
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _eval neighbor's evaluation function
     * @param _cool a cooling schedule
     * @param _cont an external continuator
     */
    moSA(Neighborhood& _neighborhood, eoEvalFunc<EOT>& _fullEval, moEval<Neighbor>& _eval, moCoolingSchedule<EOT>& _cool, moContinuator<Neighbor>& _cont):
            moLocalSearch<Neighbor>(explorer, _cont, _fullEval),
            defaultCool(0, 0, 0, 0),
            explorer(_neighborhood, _eval, defaultSolNeighborComp, _cool)
    {}

    /**
     * General constructor for a simulated annealing
     * @param _neighborhood the neighborhood
     * @param _fullEval the full evaluation function
     * @param _eval neighbor's evaluation function
     * @param _cool a cooling schedule
     * @param _comp a solution vs neighbor comparator
     * @param _cont an external continuator
     */
    moSA(Neighborhood& _neighborhood, eoEvalFunc<EOT>& _fullEval, moEval<Neighbor>& _eval, moCoolingSchedule<EOT>& _cool, moSolNeighborComparator<Neighbor>& _comp, moContinuator<Neighbor>& _cont):
            moLocalSearch<Neighbor>(explorer, _cont, _fullEval),
            defaultCool(0, 0, 0, 0),
            explorer(_neighborhood, _eval, _comp, _cool)
    {}



private:
    moTrueContinuator<Neighbor> trueCont;
    moSimpleCoolingSchedule<EOT> defaultCool;
    moSolNeighborComparator<Neighbor> defaultSolNeighborComp;
    moSAexplorer<Neighbor> explorer;
};

#endif
