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

#ifndef _edoDistrib_h
#define _edoDistrib_h

#include <eoFunctor.h>

/** @defgroup Core
 *
 * Core functors that made the basis of EDO.
 */

/** @defgroup Distributions Distributions
 *
 * A distribution is a data structure that holds sufficient informations to
 * describe a probability density function by a set of parameters.
 *
 * It is passed across EDO operators and can be updated or manipulated by them.
 */

/** Base class for distributions. This is really just an empty shell.
 *
 * @ingroup Distributions
 * @ingroup Core
 */
template < typename EOT >
class edoDistrib : public eoFunctorBase
{
public:
    //! Alias for the type
    typedef EOT EOType;

    virtual ~edoDistrib(){}
};

#endif // !_edoDistrib_h
