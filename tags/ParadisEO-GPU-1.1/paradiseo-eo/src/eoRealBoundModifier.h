// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoRealBoundModifier.h
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

#ifndef EOREALBOUNDMODIFIER_H
#define EOREALBOUNDMODIFIER_H

#include <eoFunctor.h>
#include <utils/eoRealVectorBounds.h>

/** @defgroup Bounds Bounds management
 *
 * Bounds are a set of utilities that permits to manage ranges of existence
 * for variables. For example to restrain vectors or parameters to a given domain.
 *
 * @ingroup Utilities
 */

/**
 * Abstract class for eoRealVectorBounds modifier.
 * Used to modify the bounds included into the eoRealVectorBounds object.
 *
 * @ingroup Bounds
 */
class eoRealBoundModifier: public eoBF < eoRealBaseVectorBounds &,unsigned,void > {};


/**
 * An eoRealBoundModifier that modify nothing !
 * @ingroup Bounds
 */
class eoDummyRealBoundModifier: public eoRealBoundModifier
{
public:

    eoDummyRealBoundModifier (){}

    void operator() (eoRealBaseVectorBounds & _bnds,unsigned _i)
    {
        (void)_bnds;
        (void)_i;
    }
};



/**
 * Modify an eoReal(Base)VectorBounds :
 * At iteration t, the interval I(t)=[min,max] is updated as:
 * I(t)=[min,(1-(t/Nt)^alpha)*max] where
 * - t, the current iteration, is given with an eoValueParam<unsigned>
 * - Nt is the stopping criteria <=> the total number of iterations
 * - alpha a coefficient
 *
 */
class eoExpDecayingBoundModifier: public eoRealBoundModifier
{
public:

        /**
         * Constructor
         * @param _stopCriteria - The total number of iterations
         * @param _alpha
         * @param _genCounter - An eoValueParam<unsigned> that gives the current iteration
         */
    eoExpDecayingBoundModifier (unsigned _stopCriteria,
                                double _alpha,
                                eoValueParam<unsigned> & _genCounter):
                                                stopCriteria(_stopCriteria),
                                                        alpha(_alpha),
                                                genCounter(_genCounter){}


    void operator() (eoRealBaseVectorBounds & _bnds,unsigned _i)
    {
        double newMaxBound=(1-pow((double)genCounter.value()/stopCriteria,alpha))*_bnds.maximum(_i);

        // should delete the old eoRealBounds ?
        _bnds[_i]=new eoRealInterval(_bnds.minimum(_i),std::max(_bnds.minimum(_i),newMaxBound));
    }


protected:
    unsigned stopCriteria;
    double alpha;
    eoValueParam<unsigned> & genCounter;

};

#endif/*EOREALBOUNDMODIFIER_H*/
