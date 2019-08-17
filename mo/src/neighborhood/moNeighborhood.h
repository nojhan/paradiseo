/*
  <moNeighborhood.h>
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

#ifndef _moNeighborhood_h
#define _moNeighborhood_h

#include <paradiseo/eo/eoObject.h>

/**
 * A Neighborhood
 */
template< class Neighbor >
class moNeighborhood : public eoObject
{
public:
    /**
     * Define type of a solution corresponding to Neighbor
     */
    typedef typename Neighbor::EOT EOT;

    /**
     * Default Constructor
     */
    moNeighborhood() {}

    /**
     * @return true if the neighborhood is random (default false)
     */
    virtual bool isRandom() {
        return false;
    }

    /**
     * Test if a solution has (again) a Neighbor
     * @param _solution the related solution
     * @return true if _solution has a Neighbor
     */
    virtual bool hasNeighbor(EOT & _solution) = 0 ;

    /**
     * Initialization of the neighborhood
     * @param _solution the solution to explore
     * @param _current the first neighbor
     */
    virtual void init(EOT & _solution, Neighbor & _current) = 0 ;

    /**
     * Give the next neighbor
     * @param _solution the solution to explore
     * @param _current the next neighbor
     */
    virtual void next(EOT & _solution, Neighbor & _current) = 0 ;

    /**
     * Test if there is again a neighbor
     * @param _solution the solution to explore
     * @return true if there is again a neighbor not explored
     */
    virtual bool cont(EOT & _solution) = 0 ;

    /**
     * Return the class Name
     * @return the class name as a std::string
     */
    virtual std::string className() const {
        return "moNeighborhood";
    }
};

#endif
