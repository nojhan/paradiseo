/*
(c) Thales group, 2012

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation;
    version 2 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
Contact: http://eodev.sourceforge.net

Authors:
    Benjamin Bouvier <benjamin.bouvier@gmail.com>
*/
# include "SerialArray.h"

namespace eoserial
{
    std::ostream& Array::print( std::ostream& out ) const
    {
        out << "[";
        bool first = true;
        for (ArrayChildren::const_iterator it = begin(),
                end = this->end();
                it != end;
                ++it)
        {
            if ( first )
            {
                first = false;
            } else {
                out << ", ";
            }
            (*it)->print( out );
        }
        out << "]\n";
        return out;
    }

    Array::~Array()
    {
        for (ArrayChildren::iterator it = begin(),
                end = this->end();
                it != end;
                ++it)
        {
            delete *it;
        }
    }

} // namespace eoserial
