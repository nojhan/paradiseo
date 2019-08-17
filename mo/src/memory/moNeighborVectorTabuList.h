/*
<moNeighborVectorTabuList.h>
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

#ifndef _moNeighborVectorTabuList_h
#define _moNeighborVectorTabuList_h

#include "moTabuList.h"
#include <vector>
#include <iostream>

/**
 * Tabu List of neighbors stocked in a vector
 */
template<class Neighbor >
class moNeighborVectorTabuList : public moTabuList<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT;

    /**
     * Constructor
     * @param _maxSize maximum size of the tabu list
     * @param _howlong how many iteration a move is tabu (0 -> no limits)
     */
    moNeighborVectorTabuList(unsigned int _maxSize, unsigned int _howlong) : maxSize(_maxSize), howlong(_howlong), index(0) {
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
     * add a new neighbor in the tabuList
     * @param _sol unused solution
     * @param _neighbor the current neighbor
     */
    virtual void add(EOT & _sol, Neighbor & _neighbor) {

        if (tabuList.size() < maxSize) {
            std::pair<Neighbor, unsigned int> tmp;
            tmp.first=_neighbor;
            tmp.second=howlong;
            tabuList.push_back(tmp);
        }
        else {
            tabuList[index%maxSize].first = _neighbor;
            tabuList[index%maxSize].second = howlong;
            index++;
        }
    }

    /**
     * update the tabulist
     * @param _sol unused solution
     * @param _neighbor unused neighbor
     */
    virtual void update(EOT & _sol, Neighbor & _neighbor) {
        if (howlong > 0)
            for (unsigned int i=0; i<tabuList.size(); i++)
                if (tabuList[i].second > 0)
                    tabuList[i].second--;
    }

    /**
     * check if the move is tabu
     * @param _sol unused solution
     * @param _neighbor the current neighbor
     * @return true if tabuList contains _sol
     */
    virtual bool check(EOT & _sol, Neighbor & _neighbor) {
        for (unsigned int i=0; i<tabuList.size(); i++) {
            if ((howlong > 0 && tabuList[i].second > 0 && tabuList[i].first.equals(_neighbor)) || (howlong==0 && tabuList[i].first.equals(_neighbor)))
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

    void print(){
    	std::cout << "TAbulist:" << std::endl;
    	for(int i=0; i<tabuList.size(); i++)
    		std::cout << i << ": " << tabuList[i].first.index() << std::endl;
    }


private:
    //tabu list
    std::vector< std::pair<Neighbor, unsigned int> > tabuList;
    //maximum size of the tabu list
    unsigned int maxSize;
    //how many iteration a move is tabu
    unsigned int howlong;
    //index on the tabulist
    unsigned long index;

};

#endif
