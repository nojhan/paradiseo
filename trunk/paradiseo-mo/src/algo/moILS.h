/*
<moILS.h>
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

#ifndef _moILS_h
#define _moILS_h

#include <algo/moLocalSearch.h>
#include <explorer/moILSexplorer.h>
#include <continuator/moIterContinuator.h>
#include <eoOp.h>
#include <neighborhood/moDummyNeighbor.h>
#include <perturb/moMonOpPerturb.h>
#include <acceptCrit/moAlwaysAcceptCrit.h>
#include <eval/moEval.h>
#include <eoEvalFunc.h>


/**
 * Iterated Local Search
 */
template<class Neighbor>
class moILS: public moLocalSearch<moDummyNeighbor<typename Neighbor::EOT> >
{
public:

    typedef typename Neighbor::EOT EOT;
    typedef moNeighborhood<Neighbor> Neighborhood ;

    /**
     * Basic constructor for Iterated Local Search
     * @param _ls the local search to iterates
     * @param _fullEval the full evaluation function
     * @param _op the operator used to perturb solution
     * @param _nbIteration the time limit for search
     */
    moILS(moLocalSearch<Neighbor>& _ls, eoEvalFunc<EOT>& _fullEval, eoMonOp<EOT>& _op, unsigned int _nbIteration):
            moLocalSearch<moDummyNeighbor<EOT> >(explorer, iterCont, _fullEval),
            iterCont(_nbIteration),
            defaultPerturb(_op, _fullEval),
            explorer(_ls, defaultPerturb, defaultAccept)
    {}

    /**
     * Simple constructor for Iterated Local Search
     * @param _ls the local search to iterates
     * @param _fullEval the full evaluation function
     * @param _op the operator used to perturb solution
     * @param _cont a continuator
     */
    moILS(moLocalSearch<Neighbor>& _ls, eoEvalFunc<EOT>& _fullEval, eoMonOp<EOT>& _op, moContinuator<moDummyNeighbor<EOT> >& _cont):
            moLocalSearch<moDummyNeighbor<EOT> >(explorer, _cont, _fullEval),
            iterCont(0),
            defaultPerturb(_op, _fullEval),
            explorer(_ls, defaultPerturb, defaultAccept)
    {}

    /**
     * General constructor for Iterated Local Search
     * @param _ls the local search to iterates
     * @param _fullEval the full evaluation function
     * @param _cont a continuator
     * @param _perturb a perturbation operator
     * @param _accept a acceptance criteria
     */
    moILS(moLocalSearch<Neighbor>& _ls, eoEvalFunc<EOT>& _fullEval, moContinuator<moDummyNeighbor<EOT> >& _cont, moMonOpPerturb<Neighbor>& _perturb, moAcceptanceCriterion<Neighbor>& _accept):
            moLocalSearch<moDummyNeighbor<EOT> >(explorer, _cont, _fullEval),
            iterCont(0),
            defaultPerturb(dummyOp, _fullEval),
            explorer(_ls, _perturb, _accept)
    {}

private:

    class dummmyMonOp: public eoMonOp<EOT> {
    public:
        bool operator()(EOT&) {
            return false;
        }
    }dummyOp;
    moIterContinuator<moDummyNeighbor<EOT> > iterCont;
    moMonOpPerturb<Neighbor> defaultPerturb;
    moAlwaysAcceptCrit<Neighbor> defaultAccept;
    moILSexplorer<Neighbor> explorer;
};


#endif
