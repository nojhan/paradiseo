/*
<moShiftNeighbor.h>
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

#ifndef _moShiftNeighbor_h
#define _moShiftNeighbor_h

#include <neighborhood/moIndexNeighbor.h>

/**
 * Indexed Shift Neighbor
 */
template <class EOT, class Fitness=typename EOT::Fitness>
class moShiftNeighbor: public moIndexNeighbor<EOT, Fitness>
{
public:

    using moIndexNeighbor<EOT, Fitness>::key;

    /**
     * Apply move on a solution regarding a key
     * @param _sol the solution to move
     */
    virtual void move(EOT & _sol) {
        unsigned int tmp ;
        size=_sol.size();
        translate(key+1);
        // keep the first component to change
        tmp = _sol[first];
        // shift
        if (first < second) {
            for (unsigned int i=first; i<second-1; i++)
                _sol[i] = _sol[i+1];
            // shift the first component
            _sol[second-1] = tmp;
        }
        else {  /* first > second*/
            for (unsigned int i=first; i>second; i--)
                _sol[i] = _sol[i-1];
            // shift the first component
            _sol[second] = tmp;
        }
        _sol.invalidate();
    }

    /**
     * fix two indexes regarding a key
     * @param _key the key allowing to compute the two indexes for the shift
     */
    void translate(unsigned int _key) {
        int step;
        int val = _key;
        int tmpSize = size * (size-1) / 2;
        // moves from left to right
        if (val <= tmpSize) {
            step = size - 1;
            first = 0;
            while ((val - step) > 0) {
                val = val - step;
                step--;
                first++;
            }
            second = first + val + 1;
        }
        // moves from right to left (equivalent moves are avoided)
        else {  /* val > tmpSize */
            val = val - tmpSize;
            step = size - 2;
            second = 0;
            while ((val - step) > 0) {
                val = val - step;
                step--;
                second++;
            }
            first = second + val + 1;
        }
    }

    void print() {
        std::cout << key << ": [" << first << ", " << second << "] -> " << (*this).fitness() << std::endl;
    }

private:
    unsigned int first;
    unsigned int second;
    unsigned int size;

};

#endif
