# include "Array.h"

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
