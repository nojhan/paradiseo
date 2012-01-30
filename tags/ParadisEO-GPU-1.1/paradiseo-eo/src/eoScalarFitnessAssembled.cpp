// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoScalarFitnessAssembled.cpp
// Marc Wintermantel & Oliver Koenig
// IMES-ST@ETHZ.CH
// March 2003

/*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             Marc.Schoenauer@inria.fr
             mak@dhi.dk
*/
//-----------------------------------------------------------------------------
#ifdef _MSC_VER
// to avoid long name warnings
#pragma warning(disable:4786)
#endif

#include "eoScalarFitnessAssembled.h"


// need to allocate the static variables of class eoScalarFitnessAssembledTraits
std::vector<std::string> eoScalarFitnessAssembledTraits::TermDescriptions;
