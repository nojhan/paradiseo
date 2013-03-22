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
# ifndef __EOSERIAL_UTILS_H__
# define __EOSERIAL_UTILS_H__

# include "SerialArray.h"
# include "SerialObject.h"
# include "SerialString.h"

# include "Traits.h"

# include <list>

/**
 * @file Utils.h
 *
 * @brief Contains utilities for simple serialization and deserialization.
 *
 * @todo comment new version.
 *
 * @todo encapsulate implementations.
 *
 * @todo provide more composite implementations (map<String, T>)
 */

namespace eoserial
{
    /* ***************************
     * DESERIALIZATION FUNCTIONS *
     *****************************
    These functions are useful for casting eoserial::objects into simple, primitive
    variables or into class instance which implement eoserial::Persistent.

    The model is always quite the same : 
    - the first argument is the containing object (which is a eoserial::Entity, 
    an object or an array)
    - the second argument is the key or index,
    - the last argument is the value in which we're writing.
    */

    template< class T >
    inline void unpackBasePushBack( const Entity* obj, T& container )
    {
        const Array* arr = static_cast<const Array*>( obj );
        for( auto it = arr->begin(), end = arr->end();
            it != end;
            ++it )
        {
            typename T::value_type item;
            unpackBase( *it, item );
            container.push_back( item );
        }
    }

    template< class T >
    inline void unpackBase( const Entity* obj, std::vector<T>& v )
    {
        unpackBasePushBack( obj, v );
    }

    template< class T >
    inline void unpackBase( const Entity* obj, std::list<T>& l )
    {
        unpackBasePushBack( obj, l );
    }

    template<class T, int n>
    struct UnpackImpl
    {
        void operator()( const Entity* obj, T& value )
        {
            static_cast<const String*>( obj )->deserialize( value );
        }
    };

    template<class T>
    struct UnpackImpl<T, 1>
    {
        void operator()( const Entity* obj, T& value )
        {
            value.unpack( static_cast<const Object*>(obj) );
        }
    };

    template<class T>
    inline void unpackBase( const Entity* obj, T& value )
    {
        UnpackImpl< T, IsDerivedFrom< T, Persistent >::value > impl;
        impl( obj, value );
    }

    template<class T>
    inline void unpack( const Object& obj, const std::string& key, T& value )
    {
        unpackBase( obj.find(key)->second, value );
    }

    /* *******************
     * SERIALIZATION *****
     ********************/

    template<class T, int n>
    struct PackImpl
    {
        Entity* operator()( const T& value )
        {
            std::stringstream ss;
            ss.precision(std::numeric_limits<double>::digits10 + 1);
            ss << value;
            return new String(ss.str());
        }
    };

    template<class T>
    struct PackImpl<T, 1>
    {
        Entity* operator()( const T& value )
        {
            return value.pack();
        }
    };

    template<class T>
    inline Entity* pack( const T& value );

    template<class T>
    inline Entity* packIterable( const T& container )
    {
        Array* arr = new Array;
        for( auto it = container.begin(), end = container.end();
                it != end;
                ++it )
        {
            arr->push_back( pack(*it) );
        }
        return arr;
    }

    template<class T>
    inline Entity* pack( const std::vector<T>& v )
    {
        return packIterable( v );
    }

    template<class T>
    inline Entity* pack( const std::list<T>& l )
    {
        return packIterable( l );
    }

    template<class T>
    inline Entity* pack( const T& value )
    {
        PackImpl<T, IsDerivedFrom< T, Printable >::value> impl;
        return impl( value );
    }

    // Kept for compatibility
    inline void unpackObject( const Object & obj, const std::string & key, Persistent & value )
    {
        static_cast<Object*>( obj.find( key )->second )->deserialize( value );
    }

    template< class Container, template<class> class UnpackAlgorithm >
    inline void unpackArray( const Object & obj, const std::string & key, Container & array )
    {
        static_cast<Array*>( obj.find( key )->second )->deserialize< Container, UnpackAlgorithm >( array );
    }

    template< class T >
    inline void unpack( const Array & array, unsigned int index, T & value )
    {
        static_cast<String*>( array[ index ] )->deserialize( value );
    }

    inline void unpackObject( const Array & array, unsigned int index, Persistent & value )
    {
        static_cast<Object*>( array[ index ] )->deserialize( value );
    }

    template< class Container, template<class> class UnpackAlgorithm >
    inline void unpackArray( const Array & array, unsigned int index, Container & container )
    {
        static_cast<Array*>( array[ index ] )->deserialize< Container, UnpackAlgorithm >( container );
    }

    /* *****************************
     *** SERIALIZATION FUNCTIONS ***
     *******************************
     These functions are useful for casting classic objects and 
     eoserial::Persistent objects into eoserial entities which 
     can be manipulated by the framework.
    */

    /**
     * @brief Casts a value of a stream-serializable type (i.e, which implements
     * operator <<) into a JsonString.
     *
     * This is used when serializing the objects : all primitives types should be
     * converted into strings to get more easily manipulated.
     *
     * @param value The value we're converting.
     * @return JsonString wrapper for the value.
     */
    template <typename T>
    String* make( const T & value )
    {
        std::stringstream ss;
        ss.precision(std::numeric_limits<double>::digits10 + 1);
        ss << value;
        return new String( ss.str() );
    }

    /**
     * @brief Specialization for strings : no need to convert as they're still
     * usable as strings.
     */
    template<>
    inline String* make( const std::string & value )
    {
        return new String( value );
    }

    /*
     * These functions are useful for automatically serializing STL containers into
     * eoserial arrays which could be used by the framework.
     **/

    /**
     * @brief Functor which explains how to push the value into the eoserial::Array.
     */
    template< class T >
    struct PushAlgorithm
    {
        /**
         * @brief Main operator.
         *
         * @param array The eoserial::array in which we're writing.
         * @param value The variable we are writing.
         */
        virtual void operator()( Array & array, const T & value ) = 0;
    };

    /**
     * @brief Push algorithm for primitive variables.
     *
     * This one should be used when inserting primitive (and types which implement
     * operator<<) variables.
     */
    template< class T >
    struct MakeAlgorithm : public PushAlgorithm<T>
    {
        void operator()( Array & array, const T & value )
        {
            array.push_back( make( value ) );
        }
    };

    /**
     * @brief Push algorithm for eoserial::Persistent variables.
     */
    template< class T >
    struct SerializablePushAlgorithm : public PushAlgorithm<T>
    {
        void operator()( Array & array, const T & obj )
        {
            // obj address is not saved into array.push_back.
            array.push_back( &obj );
        }
    };

    /**
     * @brief Casts a STL container (vector<int> or list<std::string>, for instance)
     * into a eoserial::Array.
     *
     * @þaram PushAlgorithm The algorithm used for inserting new element in the eoserial::Array.
     * This algorithm is directly called, so it is its own charge to invoke push_back on the 
     * eoserial::Array.
     */
    template< class Container, template<class> class PushAlgorithm >
    Array* makeArray( const Container & array )
    {
        Array* returned_array = new Array;
        typedef typename Container::const_iterator iterator;
        typedef typename Container::value_type Type;
        PushAlgorithm< Type > algo;
        for (
                iterator it = array.begin(), end = array.end();
                it != end;
                ++it)
        {
            algo( *returned_array, *it );
        }
        return returned_array;
    }
} // namespace eoserial

# endif //__EOSERIAL_UTILS_H__
