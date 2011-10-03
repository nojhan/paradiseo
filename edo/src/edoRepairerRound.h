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

#ifndef _edoRepairerRound_h
#define _edoRepairerRound_h

#include <cmath>

#include "edoRepairer.h"

/**
 *
 * @ingroup Repairers
 */
template < typename EOT >
class edoRepairerFloor : public edoRepairer<EOT>
{
public:
    virtual void operator()( EOT& sol )
    {
        for( unsigned int i=0; i < sol.size(); ++i ) {
            sol[i] = floor( sol[i] );
        }
    }
};

/**
 *
 * @ingroup Repairers
 */
template < typename EOT >
class edoRepairerCeil : public edoRepairer<EOT>
{
public:
    virtual void operator()( EOT& sol )
    {
        for( unsigned int i=0; i < sol.size(); ++i ) {
            sol[i] = ceil( sol[i] );
        }
    }
};


#endif // !_edoRepairerRound_h
