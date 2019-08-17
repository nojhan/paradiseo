/*
  <moSolNeighborComparator.h>
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

#ifndef _moSolNeighborComparator_h
#define _moSolNeighborComparator_h

#include <paradiseo/eo/EO.h>
#include <paradiseo/eo/eoFunctor.h>

#include "../neighborhood/moNeighbor.h"
#include "moComparator.h"


/**
 * Comparator of a solution and its neighbor
 */
template< class Neighbor >
class moSolNeighborComparator : public moComparator<typename Neighbor::EOT, Neighbor>
{
public:
    typedef typename Neighbor::EOT EOT ;

    /**
     * Compare two neighbors
     * @param _sol the solution
     * @param _neighbor the neighbor
     * @return true if the neighbor is better than sol
     */
    virtual bool operator()(const EOT& _sol, const Neighbor& _neighbor) {
        return (_sol.fitness() < _neighbor.fitness());
    }

    /**
     * Test the equality between two neighbors
     * @param _sol the solution
     * @param _neighbor the neighbor
     * @return true if the neighbor is equal to the solution
     */
    virtual bool equals(const EOT& _sol, const Neighbor& _neighbor) {
        return (_sol.fitness() == _neighbor.fitness());
    }

    /**
     * Return the class Name
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moSolNeighborComparator";
    }
};


#endif
