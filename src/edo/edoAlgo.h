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


#ifndef _edoAlgo_h
#define _edoAlgo_h

#include "../eo/eoAlgo.h"

/**
  @defgroup Algorithms Algorithms

  In EDO, as in EO, an algorithm is a functor that takes one or several
  solutions to an optimization problem as arguments, and iteratively modify
  them with the help of operators.It differs from a canonical EO algorithm
  because it is templatized on a edoDistrib rather than just an EOT.

  @see eoAlgo
*/

/** An EDO algorithm differs from a canonical EO algorithm because it is
 * templatized on a Distribution rather than just an EOT.
 *
 * Derivating from an eoAlgo, it should define an operator()( EOT sol )
 *
 * @ingroup Algorithms
 */
template < typename D >
class edoAlgo : public eoAlgo< typename D::EOType >
{
    //! Alias for the type
    typedef typename D::EOType EOType;

    // virtual R operator()(A1) = 0; (defined in eoUF)

public:
    virtual ~edoAlgo(){}
};

#endif // !_edoAlgo_h
