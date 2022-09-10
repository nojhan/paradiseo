/*
<moCountMoveMemory.h>
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

#ifndef _moCountMoveMemory_h
#define _moCountMoveMemory_h

#include <memory/moMemory.h>

/**
 * Count the number of move, no move and  the successive stagnation since the last Move
 */
template< class Neighbor >
class moCountMoveMemory : virtual public moMemory<Neighbor> {

public:
    typedef typename Neighbor::EOT EOT;

    /**
     * Init all the counters
     * @param _sol unused solution
     */
    void init(EOT & /*_sol*/) {
        nbMove=0;
        nbNoMove=0;
        counter=0;
    }

    /**
     * @param _sol unused solution
     * @param _neighbor unused neighbor
     */
    void add(EOT & /*_sol*/, Neighbor & /*_neighbor*/) {
        nbMove++;
        counter=0;
    }

    /**
     * @param _sol unused solution
     * @param _neighbor unused neighbor
     */
    void update(EOT & /*_sol*/, Neighbor & /*_neighbor*/) {
        nbNoMove++;
        counter++;
    }

    /**
     * ClearMemory : Reinit all the counters
     */
    void clearMemory() {
        nbMove=0;
        nbNoMove=0;
        counter=0;
    }

    /**
     * Getter of the number of move
     * @return the counter
     */
    unsigned int getNbMove() {
        return nbMove;
    }

    /**
     * Getter of the number of no move
     * @return the counter
     */
    unsigned int getNbNoMove() {
        return nbNoMove;
    }

    /**
     * Getter of the number of successive stagnation since the last Move
     * @return the counter
     */
    unsigned int getCounter() {
        return counter;
    }

    /**
     * Init counter
     */
    void initCounter() {
        counter=0;
    }

private:
    unsigned int nbMove;
    unsigned int nbNoMove;
    unsigned int counter;

};

#endif
