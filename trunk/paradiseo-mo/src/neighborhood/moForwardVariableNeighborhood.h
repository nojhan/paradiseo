/*
<moForwardVariableNeighborhood.h>
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

#ifndef _moForwardVariableNeighborhood_h
#define _moForwardVariableNeighborhood_h

#include <neighborhood/moVariableNeighborhood.h>

/**
 * A variable Neighborhood Search (VNS) in the forward manner
 */
template< class EOT >
class moForwardVariableNeighborhood : public moVariableNeighborhood<EOT>
{
public:

    typedef moNeighbor<EOT> Neighbor;

    using moVariableNeighborhood<EOT>::currentNH;
    using moVariableNeighborhood<EOT>::neighborhoodVector;

    /**
     * Construction of at least one neighborhood
     * @param _firstNH first neighborhood in the vector
     */
    moForwardVariableNeighborhood(moNeighborhood<Neighbor>& _firstNH) : moVariableNeighborhood<EOT>(_firstNH) { }

    /**
     * Return the class id.
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moForwardVariableNeighborhood";
    }

    /**
     * test if there is still some neighborhood to explore
     * @return true if there is some neighborhood to explore
     */
    virtual bool contNeighborhood() {
        return (currentNH < neighborhoodVector.size() - 1);
    }

    /**
     * put the current neighborhood on the first one
     */
    virtual void initNeighborhood() {
        currentNH = 0;
    }

    /**
     * put the current neighborhood on the next one
     */
    virtual void nextNeighborhood() {
        currentNH++;
    }

};

#endif
