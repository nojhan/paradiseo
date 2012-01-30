// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoStandardFlight.h
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

#ifndef EOSTANDARDFLIGHT_H
#define EOSTANDARDFLIGHT_H

//-----------------------------------------------------------------------------
#include <eoFlight.h>
//-----------------------------------------------------------------------------



/** Standard flight for particle swarm optimization. Derivated from abstract eoFlight,
 *   just adds the velocities to the current position of the particle
 *   and invalidates its fitness
 *
 *   @ingroup Variators
 */
template < class POT > class eoStandardFlight:public eoFlight < POT >
{

public:

    /*
        * Each element for the postion evaluation is expected to be of type PositionType.
        */
    typedef typename POT::AtomType PositionType;


    /** Constructor without bounds.
     *
     */
    eoStandardFlight ():bnds (*(new eoRealVectorNoBounds(0))){}


    /** Constructor for continuous flight with real bounds: expects a eoRealVectorBounds object for bound
     *   control.
     * @param _bounds - An eoRealVectorBounds
     */
    eoStandardFlight (eoRealVectorBounds & _bounds):bnds (_bounds){}


    /** Constructor for continuous flight with real bounds: expects a min and a
     *  max to build the same real bounds for all the positions.
     *  WARNING: _min and max MUST be double as the position type
     * @param _dim - The dimension of the bounds = the dimension of the position
     * @param _min - The lower bound to use for all the positions
     * @param _max - The upper bound to use for all the positions
     */
    eoStandardFlight (const unsigned _dim,const double & _min,const double & _max ):bnds (*(new eoRealVectorBounds(_dim,_min,_max))){}


    /**
     * Apply the standard flight to a particle : position:=position + velocity
     *  and ... invalidates the particle's fitness
     */
    void operator  () (POT & _po)
    {
        // need to resize the bounds even if there are dummy because of "isBounded" call
        bnds.adjust_size(_po.size());

        for (unsigned j = 0; j < _po.size (); j++)
        {
            PositionType newPosition;

            // tmp position
            newPosition = _po[j] + _po.velocities[j];

            /* check bounds */
            if (bnds.isMinBounded(j))
                newPosition=std::max(newPosition,bnds.minimum(j));
            if (bnds.isMaxBounded(j))
                newPosition=std::min(newPosition,bnds.maximum(j));

            _po[j]=newPosition;
        }
        // invalidate the fitness because the positions have changed
        _po.invalidate();
    }

protected:
    eoRealVectorBounds & bnds;
};




#endif /*EOSTANDARDFLIGHT_H */
