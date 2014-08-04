/*
<moSolVectorTabuList.h>
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

#ifndef _moSolVectorTabuList_h
#define _moSolVectorTabuList_h

#include "moTabuList.h"
#include <vector>
#include <iostream>

/**
 * Tabu List of solution stocked in a vector
 */
template<class Neighbor >
class moSolVectorTabuList : public moTabuList<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT;

    /**
     * Constructor
     * @param _maxSize maximum size of the tabu list
     * @param _howlong how many iteration a solution is tabu
     */
    moSolVectorTabuList(unsigned int _maxSize, unsigned int _howlong) : maxSize(_maxSize), howlong(_howlong) {
        tabuList.reserve(_maxSize);
        tabuList.resize(0);
    }

    /**
     * init the tabuList by clearing the memory
     * @param _sol the current solution
     */
    virtual void init(EOT & _sol) {
        clearMemory();
    }


    /**
     * add a new solution in the tabuList
     * @param _sol the current solution
     * @param _neighbor unused neighbor
     */
    virtual void add(EOT & _sol, Neighbor & _neighbor) {

        if (tabuList.size() < maxSize) {
            std::pair<EOT, unsigned int> tmp;
            tmp.first=_sol;
            tmp.second=howlong;
            tabuList.push_back(tmp);
        }
        else {
            tabuList[index%maxSize].first = _sol;
            tabuList[index%maxSize].second = howlong;
            index++;
        }
    }

    /**
     * update the tabulist: NOTHING TO DO
     * @param _sol the current solution
     * @param _neighbor the current neighbor (unused)
     */
    virtual void update(EOT & _sol, Neighbor & _neighbor) {
        if (howlong > 0)
            for (unsigned int i=0; i<tabuList.size(); i++)
                if (tabuList[i].second > 0)
                    tabuList[i].second--;
    }

    /**
     * check if the solution is tabu
     * @param _sol the current solution
     * @param _neighbor the current neighbor (unused)
     * @return true if tabuList contains _sol
     */
    virtual bool check(EOT & _sol, Neighbor & _neighbor) {
        EOT tmp=_sol;
        _neighbor.move(tmp);
        for (unsigned int i=0; i<tabuList.size(); i++) {
            if ((howlong > 0 && tabuList[i].second > 0 && tabuList[i].first == tmp) || (howlong==0 && tabuList[i].first==tmp))
                return true;
        }
        return false;
    }

    /**
     * clearMemory: remove all solution of the tabuList
     */
    virtual void clearMemory() {
        tabuList.resize(0);
        index = 0;
    }


private:
    //tabu list
    std::vector< std::pair<EOT, unsigned int> > tabuList;
    //maximum size of the tabu list
    unsigned int maxSize;
    //how many iteration a move is tabu
    unsigned int howlong;
    //index on the tabulist
    unsigned long index;

};

#endif
