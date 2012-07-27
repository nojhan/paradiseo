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

Copyright (C) 2010 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
    Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _edoEstimator_h
#define _edoEstimator_h

#include <eoPop.h>
#include <eoFunctor.h>

/** @defgroup Estimators Estimators
 *
 * Estimators takes an eoPop and estimates the parameters of a distributions
 * (defined as an hypothesis) from it.
 */

/** Base class for estimators.
 *
 * Estimators takes an eoPop and estimates the parameters of a distributions
 * (defined as an hypothesis) from it.
 *
 * @ingroup Estimators
 * @ingroup Core
 */
template < typename D >
class edoEstimator : public eoUF< eoPop< typename D::EOType >&, D >
{
public:
    typedef typename D::EOType EOType;

    // virtual D operator() ( eoPop< EOT >& )=0 (provided by eoUF< A1, R >)
};

#endif // !_edoEstimator_h
