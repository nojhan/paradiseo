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

#ifndef _edoStat_h
#define _edoStat_h

// eo's
#include "../../eo/eoFunctor.h"

//! edoStatBase< D >

template < typename D >
class edoStatBase : public eoUF< const D&, void >
{
public:
    // virtual void operator()( const D& ) = 0 (provided by eoUF< A1, R >)

    virtual void lastCall( const D& ) {}
    virtual std::string className() const { return "edoStatBase"; }
};

template < typename D > class edoCheckPoint;

template < typename D, typename T >
class edoStat : public eoValueParam< T >, public edoStatBase< D >
{
public:
    edoStat(T value, std::string description)
	: eoValueParam< T >(value, description)
    {}

    virtual std::string className(void) const { return "edoStat"; }

    edoStat< D, T >& addTo(edoCheckPoint< D >& cp) { cp.add(*this); return *this; }

    // TODO: edoStat< D, T >& addTo(eoMonitor& mon) { mon.add(*this); return *this; }
};


//! A parent class for any kind of distribution to dump parameter to std::string type

template < typename D >
class edoDistribStat : public edoStat< D, std::string >
{
public:
    using edoStat< D, std::string >::value;

    edoDistribStat(std::string desc)
	: edoStat< D, std::string >("", desc)
    {}
};

#endif // !_edoStat_h
