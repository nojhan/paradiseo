/*
<moVariableNeighborhood.h>
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

#ifndef _moVariableNeighborhood_h
#define _moVariableNeighborhood_h

#include <neighborhood/moNeighborhood.h>
#include <neighborhood/moNeighbor.h>
#include <neighborhood/moIndexNeighbor.h>
#include <vector>

/**
 * A vector of neighborhood for the Variable Neighborhood Search (VNS)
 */
template< class EOT >
class moVariableNeighborhood : public moNeighborhood<moNeighbor<EOT> >
{
public:

	typedef moNeighbor<EOT> Neighbor;
    /**
     * Construction of at least one neighborhood
     * @param _firstNH first neighborhood in the vector
     */
    moVariableNeighborhood(moNeighborhood<Neighbor>& _firstNH) {
			neighborhoodVector.push_back(&_firstNH);
			// the current neighborhood
			currentNH = 0;
    }

    /**
     * @return if the current neighborhood is random
     */
    virtual bool isRandom() {
        return neighborhoodVector[currentNH]->isRandom();
    }

    /**
     * Test if a solution has a Neighbor in the current neighborhood
     * @param _solution the related solution
     * @return if _solution has a Neighbor in the current neighborhood
     */
    virtual bool hasNeighbor(EOT & _solution) {
    	return neighborhoodVector[currentNH]->hasNeighbor(_solution);
    }

    /**
     * Initialization of the current neighborhood
     * @param _solution the solution to explore
     * @param _current the first neighbor in the current neighborhood
     */
    virtual void init(EOT & _solution, Neighbor & _current) {
    	neighborhoodVector[currentNH]->init(_solution, _current);
    }

    /**
     * Give the next neighbor in the current neighborhood
     * @param _solution the solution to explore
     * @param _current the next neighbor in the current neighborhood
     */
    virtual void next(EOT & _solution, Neighbor & _current) {
    	neighborhoodVector[currentNH]->next(_solution, _current);
    }

    /**
     * Test if there is again a neighbor in the current neighborhood
     * @param _solution the solution to explore
     * @return if there is still a neighbor not explored in the current neighborhood
     */
    virtual bool cont(EOT & _solution) {
    	return neighborhoodVector[currentNH]->cont(_solution);
    }

    /**
     * Return the class id.
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moVariableNeighborhood";
    }

    /**
     * to add a neighborhood in the vector
     * @param _nh the neighborhood to add at the end of the vector of neighborhood
     */
    virtual void add(moNeighborhood<Neighbor>& _nh) {
		neighborhoodVector.push_back(&_nh);
    }

    /**
     * test if there is still some neighborhood to explore
     * @return true if there is some neighborhood to explore
     */
    virtual bool contNeighborhood() = 0;

    /**
     * put the current neighborhood on the first one
     */
    virtual void initNeighborhood() = 0;

    /**
     * put the current neighborhood on the next one
     */
    virtual void nextNeighborhood() = 0;

protected:
    // the vector of neighborhoods
    std::vector<moNeighborhood<Neighbor>* > neighborhoodVector;
    // the index of the current neighborhood
    unsigned int currentNH;

};
#endif
