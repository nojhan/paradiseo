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

#ifndef _edoEstimatorAdaptiveReset_h
#define _edoEstimatorAdaptiveReset_h

#include <eoPop.h>

#include "edoEstimatorAdaptive.h"

/** An estimator that calls `reset` on the managed distribution.
 *
 * @ingroup Estimators
 */
template<class D>
class edoEstimatorAdaptiveReset : public edoEstimatorAdaptive<D>
{
public:
    typedef typename D::EOType EOType;

    edoEstimatorAdaptiveReset<D>( D& distrib ) : edoEstimatorAdaptive<D>(distrib) {}

    virtual D operator() ( eoPop<EOType>& )
    {
        this->_distrib.reset();
        return this->_distrib;
    }

};

#endif // !_edoEstimatorAdaptiveReset_h
