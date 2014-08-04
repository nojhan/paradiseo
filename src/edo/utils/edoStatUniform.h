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

#ifndef _edoStatUniform_h
#define _edoStatUniform_h

#include "edoStat.h"
#include "../edoUniform.h"

//! edoStatUniform< EOT >

template < typename EOT >
class edoStatUniform : public edoDistribStat< edoUniform< EOT > >
{
public:
    using edoDistribStat< edoUniform< EOT > >::value;

    edoStatUniform( std::string desc = "" )
	: edoDistribStat< edoUniform< EOT > >( desc )
    {}

    void operator()( const edoUniform< EOT >& distrib )
    {
	value() = "\n# ====== uniform distribution dump =====\n";

	std::ostringstream os;
	os << distrib.min() << " " << distrib.max() << std::endl;

	value() += os.str();
    }
};

#endif // !_edoStatUniform_h
