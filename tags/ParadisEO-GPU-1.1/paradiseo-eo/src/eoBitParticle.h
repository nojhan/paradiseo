// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoBitParticle.h
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

#ifndef _EOBITPARTICLE_H
#define _EOBITPARTICLE_H


#include <eoVectorParticle.h>


/** eoBitParticle: Implementation of a bit-coded particle (swarm optimization).
 *  Positions and best positions are 0 or 1 but the velocity is a vector of double.
 *
 *  @ingroup Bitstring
*/
template < class FitT> class eoBitParticle: public eoVectorParticle<FitT,bool,double>

{
public:

    eoBitParticle(unsigned size = 0, bool positions = 0,double velocities = 0.0,bool bestPositions = 0): eoVectorParticle<FitT, bool,double> (size, positions,velocities,bestPositions) {}

    virtual std::string className() const
    {
        return "eoBitParticle";
    }
};

#endif /*_EOBITPARTICLE_H*/
