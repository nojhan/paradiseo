// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// eoApply.h
// (c) Maarten Keijzer 2000
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
             mak@dhi.dk
 */
//-----------------------------------------------------------------------------

#ifndef _apply_h
#define _apply_h

#include <eoFunctor.h>
#include <vector>

/**
  Applies a unary function to a std::vector of things.
*/
template <class EOT>
void apply(eoUF<EOT&, void>& _proc, std::vector<EOT>& _pop)
{
    for (unsigned i = 0; i < _pop.size(); ++i)
    {
        _proc(_pop[i]);
    }
}

#endif
