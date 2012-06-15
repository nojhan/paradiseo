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
