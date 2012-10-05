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

#ifndef _edoStatNormalMulti_h
#define _edoStatNormalMulti_h

#include<sstream>

#include "edoStat.h"
#include "../edoNormalMulti.h"

#ifdef WITH_BOOST

#include <boost/numeric/ublas/io.hpp>

#else
#ifdef WITH_EIGEN

    // include nothing

#endif // WITH_EIGEN
#endif // WITH_BOOST

//! edoStatNormalMulti< EOT >
template < typename EOT >
class edoStatNormalMulti : public edoDistribStat< edoNormalMulti< EOT > >
{
public:
  //    typedef typename EOT::AtomType AtomType;

    using edoDistribStat< edoNormalMulti< EOT > >::value;

    edoStatNormalMulti( std::string desc = "" )
        : edoDistribStat< edoNormalMulti< EOT > >( desc )
    {}

    void operator()( const edoNormalMulti< EOT >& distrib )
    {
        value() = "\n# ====== multi normal distribution dump =====\n";

        std::ostringstream os;

        os << distrib.mean() << std::endl << std::endl << distrib.varcovar() << std::endl;

        // ublas::vector< AtomType > mean = distrib.mean();
        // std::copy(mean.begin(), mean.end(), std::ostream_iterator< std::string >( os, " " ));

        // ublas::symmetric_matrix< AtomType, ublas::lower > varcovar = distrib.varcovar();
        // std::copy(varcovar.begin(), varcovar.end(), std::ostream_iterator< std::string >( os, " " ));

        // os << std::endl;

        value() += os.str();
    }
};


#endif // !_edoStatNormalMulti_h
