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
# ifndef __EOSERIAL_SERIALIZABLE_H__
# define __EOSERIAL_SERIALIZABLE_H__

namespace eoserial
{
    class Object; // to avoid recursive inclusion with JsonObject

    /**
     * @brief Interface showing that object can be written to a eoserial type
     * (currently JSON).
     *
     * @ingroup Serialization
     */
    class Printable
    {
        public:
            /**
             * @brief Serializes the object to JSON format.
             * @return A JSON object created with new.
             */
            virtual eoserial::Object* pack() const = 0;
    };

    /**
     * @brief Interface showing that object can be eoserialized (written and read
     * from an input).
     *
     * Note : Persistent objects should have a default non-arguments constructor.
     *
     * @ingroup Serialization
     */
    class Persistent : public Printable
    {
        public:
            /**
             * @brief Loads class fields from a JSON object.
             * @param json A JSON object. Programmer doesn't have to delete it, it
             * is automatically done.
             */
            virtual void unpack(const eoserial::Object* json) = 0;
    };

} // namespace eoserial

# endif // __EOSERIAL_SERIALIZABLE_H__
