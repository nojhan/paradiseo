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

#ifndef _edoBounderBound_h
#define _edoBounderBound_h

#include "edoBounder.h"

/** A bounder that correct an incorrect variable by setting it to the min/max
 *
 * @ingroup Repairers
 */
template < typename EOT >
class edoBounderBound : public edoBounder< EOT >
{
public:
    edoBounderBound( EOT min, EOT max )
    : edoBounder< EOT >( min, max )
    {}

    void operator()( EOT& x )
    {
    unsigned int size = x.size();
    assert(size > 0);

    for (unsigned int d = 0; d < size; ++d) // browse all dimensions
        {
        if (x[d] < this->min()[d])
            {
            x[d] = this->min()[d];
            continue;
            }

        if (x[d] > this->max()[d])
            {
            x[d] = this->max()[d];
            }
        }
    }
};

#endif // !_edoBounderBound_h
