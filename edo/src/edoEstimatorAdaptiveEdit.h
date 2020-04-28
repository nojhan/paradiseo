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

#ifndef _edoEstimatorAdaptiveEdit_h
#define _edoEstimatorAdaptiveEdit_h

#include <eoPop.h>

#include "edoEstimatorAdaptive.h"

/** An estimator that change the parameters on the managed distribution.
 *
 * @ingroup Estimators
 */
template<class D, class P = typename D::EOType>
class edoEstimatorAdaptiveEdit : public edoEstimatorAdaptive<D>
{
public:
    typedef typename D::EOType EOType;

    /** Edit the given distribution's members with the given accessors.
     *
     * For example, to shift the distribution's mean toward pop's best element
     * (note the lack of parenthesis):
     * @code
     * edoEstimatorAdaptiveEdit<D> e(distrib,
     *      std::bind(&eoPop<S>::best_element, &pop),
     *      // std::bind(&D::mean, &distrib, std::placeholders::_1) // Fail to deduce templates
     *      // but you can use lambdas (even more readable):
     *      [&distrib](S center){distrib.mean(center);}
     * distrib.mean, pop.best_element);
     * @endcode
     */
    edoEstimatorAdaptiveEdit(
        D& distrib,
        std::function<P()> getter,
        std::function<void(P)> setter
    ) :
        edoEstimatorAdaptive<D>(distrib),
        _getter(getter),
        _setter(setter)
    {}

    virtual D operator()( eoPop<EOType>& )
    {
        _setter( _getter() );
        return this->_distrib;
    }

protected:
    std::function<  P ( )> _getter;
    std::function<void(P)> _setter;

};

#endif // !_edoEstimatorAdaptiveEdit_h
