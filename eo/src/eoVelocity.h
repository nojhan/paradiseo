// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoVelocity.h
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

    Contact: thomas.legrand@lifl.fr
 */
//-----------------------------------------------------------------------------

#ifndef EOVELOCITY_H
#define EOVELOCITY_H

//-----------------------------------------------------------------------------
#include <eoFunctor.h>
#include <utils/eoRNG.h>
#include <eoPop.h>
#include <eoTopology.h>
//-----------------------------------------------------------------------------

/**
 * Abstract class for velocities calculation (particle swarm optimization).
 * All the velocities must derivated from eoVelocity.
 *
 * @ingroup Variators
 */
template < class POT > class eoVelocity:public eoBF < POT &,unsigned , void >
{
public:
    /**
     * Apply the velocity computation to a whole given population.
     * Used for synchronous PSO.
     */
    virtual void apply (eoPop < POT > &_pop)
    {
        for (unsigned i = 0; i < _pop.size (); i++)
        {
            this->operator  ()(_pop[i],i);
        }

    }

    /**
     * Update the neighborhood of the given particle.
     */
    virtual void updateNeighborhood(POT & ,unsigned /*_indice*/){}


    /**
    * Apply the neighborhood with a whole population (used for distributed or synchronous PSO).
    */
    virtual void updateNeighborhood (eoPop < POT > &_pop)
    {
        for (unsigned i = 0; i < _pop.size (); i++)
        {
            updateNeighborhood(_pop[i],i);
        }

    }


   virtual eoTopology<POT> & getTopology () = 0;

};

#endif /*EOVELOCITY_H_H */
