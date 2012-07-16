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
# ifndef __EOSERIAL_ENTITY_H__
# define __EOSERIAL_ENTITY_H__

# include <iostream> // ostream


/**
 * @brief Contains all the necessary entities to serialize eo objects into JSON objects.
 *
 * Allows serialization from user objects into JSON objects, if they implement the interface
 * eoserial::Serializable or eoserial::Persistent. The following user objects can be serialized:
 * - primitive types (int, std::string, ...), in particular every type that can be written into a
 *   std::stringstream.
 * - objects which implement eoserial::Serializable.
 * - array of serializable things (primitive or serializable objects).
 *
 * @ingroup Utilities
 * @defgroup Serialization Serialization helpers
**/
namespace eoserial
{

/**
 * @brief JSON entity
 *
 * This class represents a JSON entity, which can be JSON objects,
 * strings or arrays. It is the base class for the JSON hierarchy.
 *
 * @ingroup Serialization
 */
class Entity
{
public:

    /**
     * Virtual dtor (base class).
     */
    virtual ~Entity() { /* empty */ }

    /**
     * @brief Prints the content of a JSON object into a stream.
     * @param out The stream in which we're printing.
     */
    virtual std::ostream& print( std::ostream& out ) const = 0;
};

} // namespace eoserial

# endif // __ENTITY_H__
