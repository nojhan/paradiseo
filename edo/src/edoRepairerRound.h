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

#include "edoRepairerApply.h"


/** A repairer that calls "floor" on each items of a solution
 *
 * Just a proxy to "edoRepairerApplyUnary<EOT, EOT::AtomType(EOT::AtomType)> rep( std::floor);"
 *
 * @ingroup Repairers
 */
template < typename EOT >
class edoRepairerFloor : public edoRepairerApplyUnary<EOT>
{
public:
    edoRepairerFloor() : edoRepairerApplyUnary<EOT>( std::floor ) {}
};


/** A repairer that calls "ceil" on each items of a solution
 *
 * @see edoRepairerFloor
 *
 * @ingroup Repairers
 */
template < typename EOT >
class edoRepairerCeil : public edoRepairerApplyUnary<EOT>
{
public:
    edoRepairerCeil() : edoRepairerApplyUnary<EOT>( std::ceil ) {}
};


// FIXME find a way to put this function as a member of edoRepairerRoundDecimals
template< typename ArgType >
ArgType edoRound( ArgType val, ArgType prec = 1.0 )
{ 
    return (val > 0.0) ? 
        floor(val * prec + 0.5) / prec : 
         ceil(val * prec - 0.5) / prec ; 
}

/** A repairer that round values at a given a precision. 
 *
 * e.g. if prec=0.1, 8.06 will be rounded to 8.1
 *
 * @see edoRepairerFloor
 * @see edoRepairerCeil
 *
 * @ingroup Repairers
 */
template < typename EOT >
class edoRepairerRoundDecimals : public edoRepairerApplyBinary<EOT>
{
public:
    typedef typename EOT::AtomType ArgType;

    //! Generally speaking, we expect decimals being <= 1, but it can work for higher values
    edoRepairerRoundDecimals( ArgType decimals ) : edoRepairerApplyBinary<EOT>( edoRound<ArgType>, 1 / decimals ) 
    {
        assert( decimals <= 1.0 );
        assert( 1/decimals >= 1.0 );
    }
};


/** A repairer that do a rounding around val+0.5
 *
 * @see edoRepairerRoundDecimals
 *
 * @ingroup Repairers
 */
template < typename EOT >
class edoRepairerRound : public edoRepairerRoundDecimals<EOT>
{
public:
    edoRepairerRound() : edoRepairerRoundDecimals<EOT>( 1.0 ) {}
};

#endif // !_edoRepairerRound_h
