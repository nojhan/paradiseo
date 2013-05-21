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
# ifndef __EOSERIAL_OBJECT_H__
# define __EOSERIAL_OBJECT_H__

# include <map>
# include <string>

# include "Entity.h"
# include "Serializable.h"

namespace eoserial
{
    /**
     * @brief JSON Object
     *
     * This class represents a JSON object, which is basically a dictionnary
     * of keys (strings) and values (JSON entities).
     *
     * @ingroup Serialization
     */
    class Object : public eoserial::Entity, public std::map< std::string, eoserial::Entity* >
    {
        public:
            typedef std::map<std::string, eoserial::Entity*> JsonValues;

            /**
             * @brief Adds a pair into the JSON object.
             * @param key The key associated with the eoserial object
             * @param json The JSON object as created with framework.
             */
            void add( const std::string& key, eoserial::Entity* json )
            {
                (*this)[ key ] = json;
            }

            /**
             * @brief Adds a pair into the JSON object.
             * @param key The key associated with the eoserial object
             * @param obj A JSON-serializable object
             */
            void add( const std::string& key, const eoserial::Printable* obj )
            {
                (*this)[ key ] = obj->pack();
            }

            /**
             * @brief Deserializes a Serializable class instance from this JSON object.
             * @param obj The object we want to rebuild.
             */
            void deserialize( eoserial::Persistent & obj )
            {
                obj.unpack( this );
            }

            /**
             * @brief Dtor
             */
            ~Object();

            /**
             * @brief Prints the content of a JSON object into a stream.
             */
            virtual std::ostream& print( std::ostream& out ) const;
    };

} // namespace eoserial
# endif // __EOSERIAL_OBJECT_H__

