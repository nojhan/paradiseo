/*
<moRoyalRoadIncrEval.h>
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

#ifndef _moRoyalRoadIncrEval_H
#define _moRoyalRoadIncrEval_H

#include "../../eval/moEval.h"
#include "../../../problems/eval/royalRoadEval.h"

/**
 * Incremental evaluation Function for the Royal Road problem
 */
template< class Neighbor >
class moRoyalRoadIncrEval : public moEval<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT;

    /**
     * Constructor
     * @param _rr full evaluation of the Royal Road (to have the same block size)
     */
    moRoyalRoadIncrEval(RoyalRoadEval<EOT> & _rr) : k(_rr.blockSize()) {}

    /**
     * incremental evaluation of the neighbor for the Royal Road problem
     * @param _solution the solution to move (bit string)
     * @param _neighbor the neighbor to consider (of type moBitNeigbor)
     */
    virtual void operator()(EOT & _solution, Neighbor & _neighbor) {
        // which block can be changed?
        unsigned int n = _neighbor.index() / k;

        // complete block?
        unsigned int offset = n * k;

        unsigned int j = 0;
        while (j < k && _solution[offset + j]) j++;

        if (j == k) // the block is complete, so the fitness decreases from one
            _neighbor.fitness(_solution.fitness() - 1);
        else {
            if ((_solution[_neighbor.index()] == 0) && (offset + j == _neighbor.index())) { // can the block be filled?
                j++; // next bit
                while ( j < k && _solution[offset + j]) j++;

                if (j == k) // the block can be filled, so the fitness increases from one
                    _neighbor.fitness(_solution.fitness() + 1);
                else
                    _neighbor.fitness(_solution.fitness());
            } else
                _neighbor.fitness(_solution.fitness());
        }
    }

protected:
    // size of the blocks
    unsigned int k;
};

#endif

