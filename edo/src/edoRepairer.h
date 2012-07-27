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

Copyright (C) 2011 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
    Pierre Savéant <pierre.saveant@thalesgroup.com>
*/

#ifndef _edoRepairer_h
#define _edoRepairer_h

#include <eoFunctor.h>

/** @defgroup Repairers
 *
 * A set of classes that modifies an unfeasible candidate 
 * solution so as to respect a given set of constraints and thus make a feasible
 * solution.
 */

/** The interface of a set of classes that modifies an unfeasible candidate 
 * solution so as to respect a given set of constraints and thus make a feasible
 * solution.
 *
 * @ingroup Repairers
 * @ingroup Core
 */
template < typename EOT >
class edoRepairer : public eoUF< EOT&, void >
{
public:
    // virtual void operator()( EOT& ) = 0 (provided by eoUF< A1, R >)
    virtual void operator()( EOT& ) {}
};

#endif // !_edoRepairer_h
