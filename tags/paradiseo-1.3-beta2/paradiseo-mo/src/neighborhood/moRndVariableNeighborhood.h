/*
<moRndVariableNeighborhood.h>
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

#ifndef _moRndVariableNeighborhood_h
#define _moRndVariableNeighborhood_h

#include <neighborhood/moVariableNeighborhood.h>
#include <algorithm>

/**
 * A variable Neighborhood Search (VNS) in the random manner
 */
template< class EOT, class Fitness >
class moRndVariableNeighborhood : public moVariableNeighborhood<EOT, Fitness>
{
public:
    typedef moNeighbor<EOT, Fitness> Neighbor;

    using moVariableNeighborhood<EOT, Fitness>::currentNH;
    using moVariableNeighborhood<EOT, Fitness>::neighborhoodVector;

    /**
     * Construction of at least one neighborhood
     * @param _firstNH first neighborhood in the vector
     */
    moRndVariableNeighborhood(moNeighborhood<Neighbor>& _firstNH) : moVariableNeighborhood<EOT, Fitness>(_firstNH) {
        indexVector.push_back(0);
    }

    /**
     * to add a neighborhood in the vector
     * @param _nh the neighborhood to add at the end of the vector of neighborhood
     */
    virtual void add(moNeighborhood<Neighbor>& _nh) {
        neighborhoodVector.push_back(_nh);
        indexVector.push_back(indexVector.size());
    }


    /**
     * Return the class id.
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moRndVariableNeighborhood";
    }

    /**
     * test if there is still some neighborhood to explore
     * @return true if there is some neighborhood to explore
     */
    virtual bool contNeighborhood() {
        return (index < neighborhoodVector.size() - 1);
    }

    /**
     * put the current neighborhood on the first one
     */
    virtual void initNeighborhood() {
        std::random_shuffle(indexVector.begin(), indexVector.end());
        index = 0;
        currentNH = indexVector[index];
    }

    /**
     * put the current neighborhood on the next one
     */
    virtual void nextNeighborhood() {
        index++;
        currentNH = indexVector[index];
    }

private:
    unsigned int index;
    std::vector<unsigned int> indexVector;
};

#endif
