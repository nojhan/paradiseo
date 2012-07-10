# include "String.h"

namespace eoserial
{
    std::ostream& String::print( std::ostream& out ) const
    {
        out << '"' << *this << '"';
        return out;
    }
} // namespace eoserial

