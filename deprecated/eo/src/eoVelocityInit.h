// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoVelocityInit.h
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

#ifndef EOVELOCITYINIT_H
#define EOVELOCITYINIT_H


#include <algorithm>

#include <eoOp.h>
#include <eoSTLFunctor.h>
#include <utils/eoRndGenerators.h>

/**
 @addtogroup Initializators
 @{
 */

/** Abstract class for velocities initilization of particle swarm optimization.*/
template < class POT > class eoVelocityInit:public eoInit < POT >
{
public:
    virtual std::string className (void) const
    {
        return "eoVelocityInit";
    }
};


/**
*  Provides a particle initialized thanks to the eoVelocityInit object given.
*/
template < class POT > class eoVelocityInitGenerator:public eoF < POT >
{
public:

    /** Ctor from a plain eoVelocityInit */
    eoVelocityInitGenerator (eoVelocityInit < POT > &_init):init (_init){}

    virtual POT operator  () ()
    {
        POT p;
        init (p);
        return (p);
    }

private:
    eoVelocityInit < POT > &init;
};

/**
    Initializer for fixed length velocities with a single type
*/
template < class POT > class eoVelocityInitFixedLength:public eoVelocityInit <
            POT >
{
public:

    typedef typename POT::ParticleVelocityType VelocityType;

    eoVelocityInitFixedLength (unsigned _combien,
                               eoRndGenerator < VelocityType >
                               &_generator):combien (_combien),
            generator (_generator)
    {
    }

    virtual void operator  () (POT & chrom)
    {
        chrom.resize (combien);
        std::generate (chrom.velocities.begin (), chrom.velocities.end (),
                       generator);
    }

private:
    unsigned combien;
    /// generic wrapper for eoFunctor (s), to make them have the function-pointer style copy semantics
    eoSTLF < VelocityType > generator;
};

/**
    Initializer for variable length velocitys with a single type
*/
template < class POT > class eoVelocityInitVariableLength:public eoVelocityInit <
            POT >
{
public:
    typedef typename POT::ParticleVelocityType VelocityType;

    /** Ctor from an eoVelocityInit */
    eoVelocityInitVariableLength (unsigned _minSize, unsigned _maxSize,
                                  eoVelocityInit < VelocityType >
                                  &_init):offset (_minSize),
            extent (_maxSize - _minSize), init (_init)
    {
        if (_minSize >= _maxSize)
            throw std::
            logic_error
            ("eoVelocityInitVariableLength: minSize larger or equal to maxSize");
    }


    virtual void operator  () (POT & _chrom)
    {
        _chrom.resizeVelocities (offset + rng.random (extent));
        typename std::vector < VelocityType >::iterator it;
        for (it = _chrom.velocities.begin (); it < _chrom.velocities.end (); it++)
            init (*it);
    }

    // accessor to the atom initializer (needed by operator constructs sometimes)
    eoInit < VelocityType > &atomInit ()
    {
        return init;
    }

private:
    unsigned offset;
    unsigned extent;
    eoVelocityInit < VelocityType > &init;
};

#endif /*EOVELOCITYINIT_H */

/** @} */
