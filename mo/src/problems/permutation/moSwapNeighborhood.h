/*
<moSwapNeighborhood.h>
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

#ifndef _moSwapNeighborhood_h
#define _moSwapNeighborhood_h

#include <problems/permutation/moSwapNeighbor.h>
#include <neighborhood/moNeighborhood.h>

/**
 * Swap Neighborhood
 */
template <class EOT, class Fitness=typename EOT::Fitness>
class moSwapNeighborhood : public moNeighborhood<moSwapNeighbor<EOT, Fitness> >
{
public:
    typedef moSwapNeighbor<EOT, Fitness> Neighbor;

    /**
     * @return true if there is at least an available neighbor
     */
    virtual bool hasNeighbor(EOT& _solution) {
        return (_solution.size() > 1);
    };

    /**
     * Initialization of the neighborhood
     * @param _solution the solution to explore
     * @param _current the first neighbor
     */
    virtual void init(EOT& /*_solution*/, Neighbor& _current) {
        indices.first=0;
        indices.second=1;
        _current.setIndices(0,1);
    }

    /**
     * Give the next neighbor
     * @param _solution the solution to explore
     * @param _current the next neighbor
     */
    virtual void next(EOT& _solution, Neighbor& _current) {
        if (indices.second==_solution.size()-1) {
            indices.first++;
            indices.second=indices.first+1;
        }
        else
            indices.second++;
        _current.setIndices(indices.first, indices.second);
    }

    /**
     * Test if there is again a neighbor
     * @param _solution the solution to explore
     * @return true if there is again a neighbor not explored
     */
    virtual bool cont(EOT& _solution) {
        return !((indices.first == (_solution.size()-2)) && (indices.second == (_solution.size()-1)));
    }

    /**
     * Return the class Name
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moSwapNeighborhood";
    }

private:
    std::pair<unsigned int, unsigned int> indices;

};

#endif
