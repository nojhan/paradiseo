// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoParticleBestInit.h
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

#ifndef _EOPARTICLEBESTINIT_H
#define _EOPARTICLEBESTINIT_H

//-----------------------------------------------------------------------------
#include <eoFunctor.h>
//-----------------------------------------------------------------------------

/**
 @addtogroup Initializators
 @{
 */

/**
 * Abstract class for particle best position initialization.
 */
template < class POT > class eoParticleBestInit:public eoUF < POT &, void >
{
public:

    /** Apply the initialization to a whole given population */
    virtual void apply (eoPop < POT > &_pop)
    {
        for (unsigned i = 0; i < _pop.size (); i++)
        {
            this->operator  ()(_pop[i]);
        }

    }

};

/**
 * Initializes the best positions of a particle as its current positions and set the
 * particle best fitness.
 */
template < class POT > class eoFirstIsBestInit:public eoParticleBestInit <POT>
{

public:

    /** Default CTor */
    eoFirstIsBestInit (){}

    void operator  () (POT & _po1)
    {
        //Set the bestPositions
                _po1.bestPositions = _po1 ;


        // set the fitness
        _po1.best(_po1.fitness());
    }

};

#endif /*_EOPARTICLEBESTINIT_H */

/** @} */
