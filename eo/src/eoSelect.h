/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

   -----------------------------------------------------------------------------
   eoSelect.h
   (c) Maarten Keijzer, GeNeura Team, 2000

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
 */
//-----------------------------------------------------------------------------

#ifndef _eoSelect_h
#define _eoSelect_h


//-----------------------------------------------------------------------------
#include <eoPop.h>
//-----------------------------------------------------------------------------

/**
eoSelect selects a number of individuals from the first argument and
puts it in the second. To emphasize that it should not try to enlarge
the population, the second argument is an eoPopRange, a simple struct
that holds a begin and end iterator to the population

@ingroup Core
@ingroup Selectors
*/
template<class EOT>
class eoSelect : public eoBF<const eoPop<EOT>&, eoPop<EOT>&, void>
{};

#endif
