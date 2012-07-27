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
*/

#ifndef _edoRepairerApply_h
#define _edoRepairerApply_h

#include <algorithm>

#include "edoRepairer.h"

/** Interface for applying an arbitrary unary function as a repairer on each item of the solution
 *
 * @ingroup Repairers
 */
template < typename EOT, typename F = typename EOT::AtomType(typename EOT::AtomType) >
class edoRepairerApply : public edoRepairer<EOT>
{
public:
    edoRepairerApply( F function ) : _function(function) {}

protected:
    F * _function;
};


/** Apply an arbitrary unary function as a repairer on each item of the solution
 *
 * By default, the signature of the expected function is "EOT::AtomType(EOT::AtomType)"
 *
 * @ingroup Repairers
 */
template < typename EOT, typename F = typename EOT::AtomType(typename EOT::AtomType)>
class edoRepairerApplyUnary : public edoRepairerApply<EOT,F>
{
public:
    edoRepairerApplyUnary( F function ) : edoRepairerApply<EOT,F>(function) {}

    virtual void operator()( EOT& sol )
    {
        std::transform( sol.begin(), sol.end(), sol.begin(), *(this->_function) );
        sol.invalidate();
    }
};


/** Apply an arbitrary binary function as a repairer on each item of the solution,
 * the second argument of the function being fixed and given at instanciation.
 *
 * @see edoRepairerApplyUnary
 *
 * @ingroup Repairers
 */
template < typename EOT, typename F = typename EOT::AtomType(typename EOT::AtomType, typename EOT::AtomType)>
class edoRepairerApplyBinary : public edoRepairerApply<EOT,F>
{
public:
    typedef typename EOT::AtomType ArgType;

    edoRepairerApplyBinary(   
            F function, 
            ArgType arg 
        ) : edoRepairerApply<EOT,F>(function), _arg(arg) {}

    virtual void operator()( EOT& sol )
    {
        // call the binary function on each item
        // TODO find a way to use std::transform here? Or would it be too bloated?
        for(typename EOT::iterator it = sol.begin(); it != sol.end(); ++it ) {
            *it = (*(this->_function))( *it, _arg );
        }
        sol.invalidate();
    }

protected:
    ArgType _arg;
};


#endif // !_edoRepairerApply_h

