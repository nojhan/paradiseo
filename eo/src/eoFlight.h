// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoFlight.h
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

#ifndef EOFLIGHT_H
#define EOFLIGHT_H

//-----------------------------------------------------------------------------
#include <eoFunctor.h>
#include <utils/eoRealVectorBounds.h>
//-----------------------------------------------------------------------------

/** Abstract class for particle swarm optimization flight.
* All the flights must derivated from eoFlight.
*
* @ingroup Variators
*/
template < class POT > class eoFlight:public eoUF < POT &, void >
{
public:

    /**
    * Apply the flight to a whole population.
    */
    virtual void apply (eoPop < POT > &_pop)
    {
        for (unsigned i = 0; i < _pop.size (); i++)
        {
            this->operator  ()(_pop[i]);
        }

    }
};

#endif /*EOFLIGHT_H */
