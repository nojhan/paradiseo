// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoLinearDecreasingWeightUp.h
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

#ifndef EOLINEARDECREASINGWEIGHTUP_H
#define EOLINEARDECREASINGWEIGHTUP_H

//-----------------------------------------------------------------------------
#include <eoWeightUpdater.h>
//-----------------------------------------------------------------------------


/**
 * Linear (inertia) weight updater for particle swarm optimization. Update a weight according to:
 * w(t)=(w(0)-w(Nt))*(Nt -t)/Nt + w(Nt) where
 *  t is the current generation/event
 *  Nt is the total number of generations/event
 *  w(0) is the initial weight
 *  w(Nt) is the last inertia weight
 *
 *  @ingroup Variators
 */
template <class WeightType, class StopCriteriaType> class eoLinearDecreasingWeightUp:public eoWeightUpdater<WeightType>
{
public:

    /**
     * Ctor
     */
    eoLinearDecreasingWeightUp(
        const StopCriteriaType & _stop,
        const WeightType  & _initialValue,
        const WeightType & _finalValue,
        eoValueParam<StopCriteriaType> &  _counter):
            stop(_stop),
            initialValue(_initialValue),
            finalValue(_finalValue),
            counter(_counter){}

    /**
     * Update the given weight
     */
    void operator() (WeightType & _weight)
    {
        _weight=(initialValue-finalValue)* (WeightType)(stop-counter.value())/(WeightType)stop + finalValue;

    }

protected:
    const StopCriteriaType & stop;
    const WeightType & initialValue,finalValue;
    eoValueParam<StopCriteriaType> & counter; // a counter of the number of past events (should say "generations")
};



#endif/*EOLINEARDECREASINGWEIGHTUP_H*/
