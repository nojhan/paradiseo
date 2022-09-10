/*
  <moRndWithoutReplNeighborhood.h>
  Copyright (C) DOLPHIN Project-Team, INRIA Lille - Nord Europe, 2006-2010

  Sébastien Verel, Arnaud Liefooghe, Jérémie Humeau

  This software is governed by the CeCILL license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited liability.

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

#ifndef _moRndWithoutReplNeighborhood_h
#define _moRndWithoutReplNeighborhood_h

#include <neighborhood/moIndexNeighborhood.h>
#include <neighborhood/moRndNeighborhood.h>
#include <utils/eoRNG.h>

/**
 * A Random without replacement Neighborhood
 */
template< class Neighbor >
class moRndWithoutReplNeighborhood : public moIndexNeighborhood<Neighbor>, public moRndNeighborhood<Neighbor>
{
public:

    /**
     * Define type of a solution corresponding to Neighbor
     */
    typedef typename Neighbor::EOT EOT;


    using moIndexNeighborhood<Neighbor>::neighborhoodSize;

    /**
     * Constructor
     * @param _neighborhoodSize the size of the neighborhood
     */
    moRndWithoutReplNeighborhood(unsigned int _neighborhoodSize): moIndexNeighborhood<Neighbor>(_neighborhoodSize), maxIndex(0) {
        for (unsigned int i=0; i < neighborhoodSize; i++)
            indexVector.push_back(i);
    }

    /**
     * Test if it exist a neighbor
     * @param _solution the solution to explore
     * @return true if the neighborhood was not empty
     */
    virtual bool hasNeighbor(EOT& /*_solution*/) {
        return neighborhoodSize > 0;
    }

    /**
     * Initialization of the neighborhood
     * @param _solution the solution to explore
     * @param _neighbor the first neighbor
     */
    virtual void init(EOT & _solution, Neighbor & _neighbor) {
        unsigned int i, tmp;
        maxIndex = neighborhoodSize ;
        i = rng.random(maxIndex);
        _neighbor.index(_solution, indexVector[i]);
        tmp=indexVector[i];
        indexVector[i]=indexVector[maxIndex-1];
        indexVector[maxIndex-1]=tmp;
        maxIndex--;
    }

    /**
     * Give the next neighbor
     * @param _solution the solution to explore
     * @param _neighbor the next neighbor
     */
    virtual void next(EOT & _solution, Neighbor & _neighbor) {
        unsigned int i, tmp;
        i = rng.random(maxIndex);
        _neighbor.index(_solution, indexVector[i]);
        tmp=indexVector[i];
        indexVector[i]=indexVector[maxIndex-1];
        indexVector[maxIndex-1]=tmp;
        maxIndex--;
    }

    /**
     * test if all neighbors are explore or not,if false, there is no neighbor left to explore
     * maxIndex is number of neighbors
     * @param _solution the solution to explore
     * @return true if there is again a neighbor to explore
     */
    virtual bool cont(EOT & /*_solution*/) {
        return (maxIndex > 0) ;
    }

    /**
     * Getter
     * @return the position in the Neighborhood
     */
    unsigned int position() {
        return indexVector[maxIndex];
    }

    /**
     * Return the class Name
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moRndWithoutReplNeighborhood";
    }

protected:
    unsigned int maxIndex;
    std::vector<unsigned int> indexVector;
};

#endif
