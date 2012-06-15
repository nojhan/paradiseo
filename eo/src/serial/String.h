# ifndef __EOSERIAL_STRING_H__
# define __EOSERIAL_STRING_H__

# include <string>
# include <sstream>

# include "Entity.h"

namespace eoserial
{

/**
 * @brief JSON String
 *
 * Wrapper for string, so as to be used as a JSON object.
 */
class String : public eoserial::Entity, public std::string
{
    public:

        /**
         * @brief Default ctor.
         * @param str The string we want to wrap.
         */
        String( const std::string& str ) : std::string( str ) {}

        /**
         * @brief Ctor used only on parsing.
         */
        String( ) {}

        /**
         * @brief Prints out the string.
         */
        virtual std::ostream& print( std::ostream& out ) const;

        /**
         * @brief Deserializes the current String into a given primitive type value.
         * @param value The value in which we're writing.
         */
        template<class T>
        inline void deserialize( T & value );

    protected:
        // Copy and reaffectation are forbidden
        explicit String( const String& _ );
        String& operator=( const String& _ );
};

/**
 * @brief Casts a eoserial::String into a primitive value, or in a type which at
 * least overload operator>>.
 *
 * @param value A reference to the variable we're writing into.
 *
 * It's not necessary to specify the variable type, which can be infered by compiler when
 * invoking.
 */
template<class T>
inline void String::deserialize( T & value )
{
    std::stringstream ss;
    ss << *this;
    ss >> value;
}

/**
 * @brief Specialization for strings, which don't need to be converted through
 * a stringstream.
 */
template<>
inline void String::deserialize( std::string & value )
{
    value = *this;
}

} // namespace eoserial

# endif // __EOSERIAL_STRING_H__
