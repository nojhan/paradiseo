// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoTopology.h
// (c) OPAC 2007
/*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: thomas.legrand@inria.fr
             clive.canape@inria.fr
 */
//-----------------------------------------------------------------------------

#ifndef EOTOPOLOGY_H_
#define EOTOPOLOGY_H_

//-----------------------------------------------------------------------------
#include <eoNeighborhood.h>
//-----------------------------------------------------------------------------


/**
 * Defines the interface of a swarm topology. Can be static (usually the case for the social topologies)
 * or dynamic. It's the same interface for both social and physical topologies. ("social" topology means
 * social-neighborhood-based toplogy and so on ...)
 *
 * @ingroup Selectors
 * @ingroup Core
 */
template < class POT > class eoTopology:public eoPop < POT >
{
public:

    /**
     * Builds the neighborhoods contained in the topology.
     */
    virtual void setup(const eoPop<POT> &)=0;

    /**
     * Updates the neighborhood of the given particle and its indice in the population
     */
    virtual void updateNeighborhood(POT & ,unsigned)=0;


    /**
      * Updates the neighborhood of the given particle thanks to a whole population (used for distributed or synchronous PSO)
      */
    virtual void updateNeighborhood(eoPop < POT > &_pop)
    {
        for (unsigned i = 0; i < _pop.size (); i++)
            updateNeighborhood(_pop[i],i);
    }


    /**
     * Builds the neighborhoods contained in the topology.
     */
    virtual POT & best (unsigned ) = 0;


        /*
         * Return the global best of the topology
         */
    virtual POT & globalBest() = 0;


    /**
     * Prints the neighborhoods contained in the topology.
     */
    virtual void printOn() = 0;
};


#endif /*EOTOPOLOGY_H_ */
