/*
<moBestImprAspiration.h>
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


#ifndef _moBestImprAspiration_h
#define _moBestImprAspiration_h

#include <memory/moAspiration.h>

/**
 * Aspiration criteria accepts a solution better than the best so far
 */
template< class Neighbor >
class moBestImprAspiration : public moAspiration<Neighbor>
{
public:

    typedef typename Neighbor::EOT EOT;

    /**
     * init the best solution
     * @param _sol the best solution found
     */
    void init(EOT & _sol) {
        bestFoundSoFar = _sol;
    }

    /**
     * update the "bestFoundSoFar" if a best solution is found
     * @param _sol a solution
     * @param _neighbor a neighbor
     */
    void update(EOT & _sol, Neighbor & /*_neighbor*/) {
        if (bestFoundSoFar.fitness() < _sol.fitness())
            bestFoundSoFar = _sol;
    }

    /**
     * Test the tabu feature of the neighbor:
     * test if the neighbor's fitness is better than the "bestFoundSoFar" fitness
     * @param _sol a solution
     * @param _neighbor a neighbor
     * @return true if _neighbor fitness is better than the "bestFoundSoFar"
     */
    bool operator()(EOT & /*_sol*/, Neighbor & _neighbor) {
        return (bestFoundSoFar.fitness() < _neighbor.fitness());
    }

    /**
     * Getter
     * @return a reference on the best found so far solution
     */
    EOT& getBest() {
        return bestFoundSoFar;
    }

private:
    EOT bestFoundSoFar;
};

#endif
