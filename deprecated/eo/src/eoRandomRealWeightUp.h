// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRandomRealWeightUp.h
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

#ifndef EORANDOMREALWEIGHTUP_H
#define EORANDOMREALWEIGHTUP_H

//-----------------------------------------------------------------------------
#include <eoWeightUpdater.h>
#include <utils/eoRNG.h>
//-----------------------------------------------------------------------------

/**
 * Update an inertia weight by assigning it an (uniform) random value.
 * The weight is a basic feature to evaluate the velocity of a particle in
 * swarm optimization.
 *
 * @ingroup Variators
 */
class eoRandomRealWeightUp:public eoWeightUpdater<double>
{
public:

    /**
     * Default constructor
     * @param _min - The minimum bound for the weight
     * @param _max - The maximum bound for the weight
     */
    eoRandomRealWeightUp(
        double  _min,
        double  _max
    ):min(_min),max(_max)
    {
        // consistency check
        if (min > max)
        {
            std::string s;
            s.append (" min > max in eoRandomRealWeightUp");
            throw std::runtime_error (s);
        }
    }

    /**
     * Generate an real random number in [min,max] and assign it to _weight
     * @param _weight - The assigned (real) weight
     */
    void operator() (double & _weight)
    {
        _weight=rng.uniform(max-min)+min;
    }


protected:
    double min,max;

};



#endif/*EORANDOMREALWEIGHTUP_H*/
