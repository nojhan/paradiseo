// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoPSO.h
// (c) OPAC 2007
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

    Contact: thomas.legrand@lifl.fr
 */
//-----------------------------------------------------------------------------

#ifndef _EOPSO_H
#define _EOPSO_H

//-----------------------------------------------------------------------------
#include <eoAlgo.h>
//-----------------------------------------------------------------------------

/**
    This is a generic class for particle swarm algorithms. There
    is only one operator defined, which takes a population and does stuff to
    it. It needn't be a complete algorithm, can be also a step of an
    algorithm. Only used for mono-objective cases.

    @ingroup Algorithms
*/
template < class POT > class eoPSO:public eoAlgo < POT >{};

#endif /*_EOPSO_H*/
