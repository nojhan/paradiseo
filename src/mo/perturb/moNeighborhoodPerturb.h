/*
<moNeighborhoodPerturb.h>
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

#ifndef _moNeighborhoodPerturb_h
#define _moNeighborhoodPerturb_h

#include "../eval/moEval.h"
#include "moPerturbation.h"
#include "../neighborhood/moNeighborhood.h"

/**
 * Neighborhood Perturbation: explore the neighborhood to perturb the solution (the neighborhood could be different as the one used in the Local Search)
 */
template< class Neighbor, class OtherNeighbor >
class moNeighborhoodPerturb : public moPerturbation<Neighbor> {

public:
    typedef typename Neighbor::EOT EOT;
    typedef moNeighborhood<OtherNeighbor> OtherNH;

    /**
     * Constructor
     * @param _otherNeighborhood a neighborhood
     * @param _eval an Evaluation Function
     */
    moNeighborhoodPerturb(OtherNH& _otherNeighborhood, moEval<OtherNeighbor>& _eval): otherNeighborhood(_otherNeighborhood), eval(_eval) {}

    /**
     * Apply move on the solution
     * @param _solution the current solution
     * @return true
     */
    virtual bool operator()(EOT& _solution) {
        if (otherNeighborhood.hasNeighbor(_solution)) {
            eval(_solution, current);
            current.move(_solution);
            _solution.fitness(current.fitness());
        }
        return true;
    }

    /**
     * Init the neighborhood
     * @param _sol the current solution
     */
    virtual void init(EOT & _sol) {
        if (otherNeighborhood.hasNeighbor(_sol))
            otherNeighborhood.init(_sol, current);
    }

    /**
     * ReInit the neighborhood because a move was done
     * @param _sol the current solution
     * @param _neighbor unused neighbor (always empty)
     */
    virtual void add(EOT & _sol, Neighbor & _neighbor) {
        (*this).init(_sol);
    }

    /**
     * Explore another neighbor because no move was done
     * @param _sol the current solution
     * @param _neighbor unused neighbor (always empty)
     */
    virtual void update(EOT & _sol, Neighbor & _neighbor) {
        if (otherNeighborhood.cont(_sol))
            otherNeighborhood.next(_sol, current);
        else
            (*this).init(_sol);
    }

    /**
     * NOTHING TO DO
     */
    virtual void clearMemory() {}

private:
    OtherNH& otherNeighborhood;
    moEval<OtherNeighbor>& eval;
    OtherNeighbor current;
};

#endif
