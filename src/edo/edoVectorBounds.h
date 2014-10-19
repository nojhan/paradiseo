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

#ifndef _edoVectorBounds_h
#define _edoVectorBounds_h

/** A class that holds min and max bounds vectors.
 */
template < typename EOT >
class edoVectorBounds
{
public:
    edoVectorBounds(EOT min, EOT max)
        : _min(min), _max(max)
    {
        assert(_min.size() > 0);
        assert(_min.size() == _max.size());
    }

    EOT min(){return _min;}
    EOT max(){return _max;}

    unsigned int size()
    {
        assert(_min.size() == _max.size());
        return _min.size();
    }

private:
    EOT _min;
    EOT _max;
};

#endif // !_edoVectorBounds_h
