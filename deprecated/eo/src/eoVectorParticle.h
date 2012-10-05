// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoVectorParticle.h
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

#ifndef _EOVECTORPARTICLE_H
#define _EOVECTORPARTICLE_H

#include <PO.h>


/**
 @addtogroup Representations
 @{
*/

/**
 * Main class for particle representation of particle swarm optimization. The positions, velocities and the best positions
 * associated to the  particle are stored as vectors. Inheriting from PO and std::vector,
 * tree templates arguments are required: the fitness type (which is also the type of the
 * particle's best fitness), the position type and the velocity type.
 */
template < class FitT, class PositionType, class VelocityType > class eoVectorParticle:public PO < FitT >,
            public std::vector <
            PositionType >
{

public:

    using PO < FitT >::invalidate;
    using
    std::vector <
    PositionType >::operator[];
    using
    std::vector <
    PositionType >::begin;
    using
    std::vector <
    PositionType >::end;
    using
    std::vector <
    PositionType >::size;

    typedef PositionType AtomType;
    typedef VelocityType ParticleVelocityType;


    /** Default constructor.
    *  @param _size Length of the tree vectors (we expect the same size), default is 0
    *  @param _position
    *  @param _velocity
    *  @param _bestPositions
    */
    eoVectorParticle (unsigned _size = 0,PositionType _position = PositionType (), VelocityType _velocity = VelocityType (), PositionType _bestPositions = PositionType ()):PO < FitT > (),std::vector < PositionType > (_size, _position), bestPositions (_size, _bestPositions), velocities (_size,
                    _velocity)
    {
    }


    // we can't have a Ctor from a std::vector, it would create ambiguity
    //  with the copy Ctor
    void
    position (const std::vector < PositionType > &_v)
    {
        if (_v.size () != size ())	// safety check
        {
            if (size ())		// NOT an initial empty std::vector
                eo::log << eo::warnings <<
                "Warning: Changing position size in eoVectorParticle assignation"
                << std::endl;
            resize (_v.size ());
        }

        std::copy (_v.begin (), _v.end (), begin ());
        invalidate ();
    }

    /** Resize the tree vectors of the particle: positions, velocities and bestPositions
    * @param _size The new size for positions, velocities and bestPositions
    */
    void
    resize (unsigned _size)
    {
        std::vector < PositionType >::resize (_size);
        bestPositions.resize (_size);
        velocities.resize (_size);
    }


    /** Resize the best positions.
       * @param _size The new size for the best positions.
       */
    void
    resizeBestPositions (unsigned _size)
    {
        bestPositions.resize (_size);
    }


    /** Resize the velocities.
    * @param _size The new size for the velocities.
    */
    void
    resizeVelocities (unsigned _size)
    {
        velocities.resize (_size);
    }

    /// to avoid conflicts between EA and PSO
    bool operator<(const eoVectorParticle<FitT, PositionType, VelocityType >& _eo) const
        {
                if (_eo.best() > this->best())
                        return true;
                else
                        return false;
        }

    /**
    * Print-on a vector-particle
    */
    virtual void printOn(std::ostream& os) const
    {
        PO<FitT>::printOn(os);
        os << ' ';
        os << size() << ' ';
        std::copy(bestPositions.begin(), bestPositions.end(), std::ostream_iterator<AtomType>(os, " "));
     }

    /* public attributes */
    std::vector < PositionType > bestPositions;
    std::vector < ParticleVelocityType > velocities;

};

#endif /*_EOVECTORPARTICLE_H*/
/** @} */
