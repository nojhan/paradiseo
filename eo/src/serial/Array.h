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
# ifndef __EOSERIAL_ARRAY_H__
# define __EOSERIAL_ARRAY_H__

# include <vector>

# include "Entity.h"
# include "Serializable.h"
# include "Object.h"

namespace eoserial
{

    // Forward declaration for below declarations.
    class Array;

    /*
     * Declarations of functions present in Utils.h
     * These are put here to avoid instead of including the file Utils.h, which would
     * cause a circular inclusion.
     */

    template< class T >
        void unpack( const Array & array, unsigned int index, T & value );

    void unpackObject( const Array & array, unsigned int index, Persistent & value );

    template< class Container, template<class> class UnpackAlgorithm >
        void unpackArray( const Array & array, unsigned int index, Container & container );

    /**
     * @brief Represents a JSON array.
     *
     * Wrapper for an array, so as to be used as a JSON object.
     *
     * @ingroup Serialization
     */
    class Array : public eoserial::Entity, public std::vector< eoserial::Entity* >
    {
        protected:
            typedef std::vector< eoserial::Entity* > ArrayChildren;

        public:
            /**
             * @brief Adds the serializable object as a JSON object.
             * @param obj Object which implemnets JsonSerializable.
             */
            void push_back( const eoserial::Printable* obj )
            {
                ArrayChildren::push_back( obj->pack() );
            }

            /**
             * @brief Proxy for vector::push_back.
             */
            void push_back( eoserial::Entity* json )
            {
                ArrayChildren::push_back( json );
            }

            /**
             * @brief Prints the JSON array into the given stream.
             * @param out The stream
             */
            virtual std::ostream& print( std::ostream& out ) const;

            /**
             * @brief Dtor
             */
            ~Array();

            /*
             * The following parts allows the user to automatically deserialize an eoserial::Array into a
             * standard container, by giving the algorithm which will be used to deserialize contained entities.
             */

            /**
             * @brief Functor which determines how to retrieve the real value contained in a eoserial::Entity at
             * a given place.
             *
             * It will be applied for each contained variable in the array.
             */
            template<class Container>
                struct BaseAlgorithm
                {
                    /**
                     * @brief Main operator.
                     *
                     * @param array The eoserial::Array from which we're reading.
                     * @param i The index of the contained value.
                     * @param container The standard (STL) container in which we'll push back the read value.
                     */
                    virtual void operator()( const eoserial::Array& array, unsigned int i, Container & container ) const = 0;
                };

            /**
             * @brief BaseAlgorithm for retrieving primitive variables.
             *
             * This one should be used to retrieve primitive (and types which implement operator>>) variables, for instance
             * int, double, std::string, etc...
             */
            template<typename C>
                struct UnpackAlgorithm : public BaseAlgorithm<C>
            {
                void operator()( const eoserial::Array& array, unsigned int i, C & container ) const
                {
                    typename C::value_type t;
                    unpack( array, i, t );
                    container.push_back( t );
                }
            };

            /**
             * @brief BaseAlgorithm for retrieving eoserial::Persistent objects.
             *
             * This one should be used to retrieve objects which implement eoserial::Persistent.
             */
            template<typename C>
                struct UnpackObjectAlgorithm : public BaseAlgorithm<C>
            {
                void operator()( const eoserial::Array& array, unsigned int i, C & container ) const
                {
                    typename C::value_type t;
                    unpackObject( array, i, t );
                    container.push_back( t );
                }
            };

            /**
             * @brief General algorithm for array deserialization.
             *
             * Applies the BaseAlgorithm to each contained variable in the eoserial::Array.
             */
            template<class Container, template<class T> class UnpackAlgorithm>
                inline void deserialize( Container & array )
                {
                    UnpackAlgorithm< Container > algo;
                    for( unsigned int i = 0, size = this->size();
                            i < size;
                            ++i)
                    {
                        algo( *this, i, array );
                    }
                }
    };

} // namespace eoserial

# endif // __EOSERIAL_ARRAY_H__

