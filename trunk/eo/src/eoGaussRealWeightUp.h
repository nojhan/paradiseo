// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoGaussRealWeightUp.h
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

#ifndef EOGAUSSREALWEIGHTUP_H
#define EOGAUSSREALWEIGHTUP_H

//-----------------------------------------------------------------------------
#include <eoWeightUpdater.h>
#include <utils/eoRNG.h>
//-----------------------------------------------------------------------------


/**
 * Update an inertia weight by assigning it a Gaussian randomized value
 * (used for the velocity in particle swarm optimization).
 *
 * @ingroup Variators
 */
class eoGaussRealWeightUp:public eoWeightUpdater<double>
{
public:

    /**
     * Default constructor
     * @param _mean - Mean for Gaussian distribution
     * @param _stdev - Standard deviation for Gaussian distribution
     */
    eoGaussRealWeightUp(
        double  _mean=0,
        double  _stdev=1.0
    ):mean(_mean),stdev(_stdev){}

    /**
     * Assign Gaussian deviation  to _weight
     * @param _weight - The modified weight as a double
     */
    void operator() (double & _weight)
    {
        _weight=rng.normal(mean,stdev);
    }


protected:
    double mean,stdev;

};



#endif/*EOGAUSSREALWEIGHTUP_H*/
