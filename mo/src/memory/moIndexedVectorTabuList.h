/*
<moIndexedVectorTabuList.h>
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

#ifndef _moIndexedVectorTabuList_h
#define _moIndexedVectorTabuList_h

#include <utils/eoRndGenerators.h>
#include <memory/moTabuList.h>
#include <vector>
#include <iostream>

/**
 *
 * Tabu List of indexed neighbors save in a vector
 * each neighbor can not used during howlong iterations
 *
 * The tabu tenure could be random between two bounds 
 * such as in robust tabu search
 *
 */
template<class Neighbor >
class moIndexedVectorTabuList : public moTabuList<Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT;

    /**
     * Constructor
     * @param _maxSize maximum size of the tabu list
     * @param _howlong how many iteration a move is tabu
     */
    moIndexedVectorTabuList(unsigned int _maxSize, unsigned int _howlong) : maxSize(_maxSize), howlong(_howlong), robust(false) {
        tabuList.resize(_maxSize);
    }

    /**
     * Constructor
     * @param _maxSize maximum size of the tabu list
     * @param _howlongMin minimal number of iterations during a move is tabu
     * @param _howlongMax maximal number of iterations during a move is tabu
     */
 moIndexedVectorTabuList(unsigned int _maxSize, unsigned int _howlongMin, unsigned int _howlongMax) : maxSize(_maxSize), howlongMin(_howlongMin), howlongMax(_howlongMax), robust(true) {
        tabuList.resize(_maxSize);
    }

    /**
     * init the tabuList by clearing the memory
     * @param _sol the current solution
     */
    virtual void init(EOT & /*_sol*/) {
        clearMemory();
    }


    /**
     * add a new neighbor in the tabuList
     * @param _sol unused solution
     * @param _neighbor the current neighbor
     */
    virtual void add(EOT & /*_sol*/, Neighbor & _neighbor) {
      if (_neighbor.index() < maxSize) {
	if (robust)
	  // random value between min and max
	  howlong = howlongMin + rng.random(howlongMax - howlongMin);

	tabuList[_neighbor.index()] = howlong;
      }
    }

    /**
     * update the tabulist by decreasing the number of tabu iteration
     * @param _sol unused solution
     * @param _neighbor unused neighbor
     */
    virtual void update(EOT & /*_sol*/, Neighbor & /*_neighbor*/) {
      for (unsigned int i = 0; i < maxSize; i++)
	if (tabuList[i] > 0)
	  tabuList[i]--;
    }

    /**
     * check if the move is tabu
     * @param _sol unused solution
     * @param _neighbor the current neighbor
     * @return true if tabuList contains _sol
     */
    virtual bool check(EOT & /*_sol*/, Neighbor & _neighbor) {
      return (tabuList[_neighbor.index()] > 0);
    }

    /**
     * clearMemory: remove all solution of the tabuList
     */
    virtual void clearMemory() {
      for (unsigned int i = 0; i < maxSize; i++)
	tabuList[i] = 0;
    }

    void print(){
    	std::cout << "Tabulist:" << std::endl;
    	for(int i=0; i<tabuList.size(); i++)
	  std::cout << i << ": " << tabuList[i] << std::endl;
    }


protected:
    //tabu list
    std::vector< unsigned int > tabuList;
    //maximum size of the tabu list
    unsigned int maxSize;
    //how many iteration a move is tabu
    unsigned int howlong;
    // Minimum number of iterations during a move is tabu
    unsigned int howlongMin;
    // Maximum number of iterations during a move is tabu
    unsigned int howlongMax;
    // true: robust tabu search way
    bool robust;
};

#endif
