// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/*

(c) Thales group, 2010

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation;
    version 2 of the license.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Contact: http://eodev.sourceforge.net

Authors:
     Caner Candan <caner.candan@thalesgroup.com>

*/

#include "eoParallel.h"

eoParallel::eoParallel()
    : _isEnabled( false, "parallelize-loop", "Enable memory shared parallelization into evaluation's loops", '\0' ),
      _isDynamic( false, "parallelize-dynamic", "Enable dynamic memory shared parallelization", '\0' ),
      _prefix( "results", "parallelize-prefix", "Enable dynamic memory shared parallelization", '\0' )
{}

std::string eoParallel::className() const
{
    return "eoParallel";
}

std::string eoParallel::prefix() const
{
    std::string value( _prefix.value() );

    if ( _isEnabled.value() )
	{
	    if ( _isDynamic.value() )
		{
		    value += "_dynamic.out";
		}
	    else
		{
		    value += "_parallel.out";
		}
	}
    else
	{
	    value += "_sequential.out";
	}

    return value;
}

void eoParallel::_createParameters( eoParser& parser )
{
    std::string section("Parallelization");
    parser.processParam( _isEnabled, section );
    parser.processParam( _isDynamic, section );
    parser.processParam( _prefix, section );
}

void make_parallel(eoParser& parser)
{
    eo::parallel._createParameters( parser );
}

eoParallel eo::parallel;
