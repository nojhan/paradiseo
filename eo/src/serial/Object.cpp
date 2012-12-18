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
# include "Object.h"

using namespace eoserial;

namespace eoserial
{
    std::ostream& Object::print( std::ostream& out ) const
    {
        out << '{';
        bool first = true;
        for(JsonValues::const_iterator it = begin(), end = this->end();
                it != end;
                ++it)
        {
            if ( first )
            {
                first = false;
            } else {
                out << ", ";
            }

            out << '"' << it->first << "\":";   // key
            it->second->print( out );           // value
        }
        out << "}\n";
        return out;
    }

    Object::~Object()
    {
        for(JsonValues::iterator it = begin(), end = this->end();
                it != end;
                ++it)
        {
            delete it->second;
        }
    }

} // namespace eoserial
