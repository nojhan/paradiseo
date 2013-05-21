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
# ifndef __EOSERIAL_STRING_H__
# define __EOSERIAL_STRING_H__

# include <string>
# include <sstream>
# include <limits>

# include "Entity.h"

namespace eoserial
{
    /**
     * @brief JSON String
     *
     * Wrapper for string, so as to be used as a JSON object.
     *
     * @ingroup Serialization
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
             * @brief Ctor used only when parsing.
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
                inline void deserialize( T & value ) const;

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
        inline void String::deserialize( T & value ) const
        {
            std::stringstream ss;
            ss.precision(std::numeric_limits<double>::digits10 + 1);
            ss << *this;
            ss >> value;
        }

    /**
     * @brief Specialization for strings, which don't need to be converted through
     * a stringstream.
     */
    template<>
        inline void String::deserialize( std::string & value ) const
        {
            value = *this;
        }

} // namespace eoserial

# endif // __EOSERIAL_STRING_H__
