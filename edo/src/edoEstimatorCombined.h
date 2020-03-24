/*
The Evolving Distribution Objects framework (EDO) is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own estimation of distribution algorithms.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (C) 2020 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
*/

#ifndef _edoEstimatorAdaptiveCombined_h
#define _edoEstimatorAdaptiveCombined_h

#include <vector>

#include <eoPop.h>

#include "edoEstimatorAdaptive.h"

/** An estimator that calls several ordered estimators, stateful version.
 *
 * @ingroup Estimators
 */
template<class  D>
class edoEstimatorCombinedAdaptive : public edoEstimatorAdaptive<D>, public std::vector<edoEstimator<D>*>
{
public:
    typedef typename D::EOType EOType;

    edoEstimatorCombinedAdaptive<D>( D& distrib, edoEstimator<D>& estim) :
        edoEstimatorAdaptive<D>(distrib),
        std::vector<edoEstimator<D>*>(1,&estim)
    {}

    edoEstimatorCombinedAdaptive<D>( D& distrib, std::vector<edoEstimator<D>*> estims) :
        edoEstimatorAdaptive<D>(distrib),
        std::vector<edoEstimator<D>*>(estims)
    {}

    void add( edoEstimator<D>& estim )
    {
        this->push_back(&estim);
    }

    virtual D operator()( eoPop<EOType>& pop )
    {
        for(edoEstimator<D>* pestim : *this) {
            this->_distrib = (*pestim)( pop );
        }
        return this->_distrib;
    }

};

/** An estimator that calls several ordered estimators, stateless version.
 *
 * @ingroup Estimators
 */
template<class D>
class edoEstimatorCombinedStateless : public edoEstimatorCombinedAdaptive<D>
{
public:
    typedef typename D::EOType EOType;

    edoEstimatorCombinedStateless<D>( edoEstimator<D>& estim ) :
        edoEstimatorCombinedAdaptive<D>(*(new D), estim)
    {}

    edoEstimatorCombinedStateless<D>( std::vector<edoEstimator<D>*> estims) :
        edoEstimatorCombinedAdaptive<D>(*(new D), estims)
    {}

    virtual D operator()( eoPop<EOType>& pop )
    {
        delete &(this->_distrib);
        this->_distrib = *(new D);
        return edoEstimatorCombinedAdaptive<D>::operator()(pop);
    }

    ~edoEstimatorCombinedStateless()
    {
        delete &(this->_distrib);
    }

};

#endif // !_edoEstimatorAdaptiveCombined_h
