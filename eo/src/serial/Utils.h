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
# include <map>

/**
 * @file Utils.h
 *
 * @brief Contains utilities for simple serialization and deserialization.
 *
 * @todo encapsulate implementations.
 *
 * Example
 *
 * @code
# include <vector>
# include <string>
# include <iostream>

# include "eoSerial.h"

struct SimpleObject: public eoserial::Persistent
{
    public:

    SimpleObject( int v ) : value(v)
    {
        // empty
    }

    eoserial::Object* pack() const
    {
        eoserial::Object* obj = new eoserial::Object;
        (*obj)["value"] = eoserial::pack( value );
        return obj;
    }

    void unpack( const eoserial::Object* json )
    {
        eoserial::unpack( *json, "value", value );
    }

    int value;
};

int main()
{
    eoserial::Object o;

    std::cout << "packing..." << std::endl;
    // directly pack raw types
    o["long"] = eoserial::pack(123456L);
    o["bool"] = eoserial::pack(true);
    o["double"] = eoserial::pack(3.141592653);
    o["float"] = eoserial::pack(3.141592653f);

    std::string str = "Hello, world!";
    o["str"] = eoserial::pack( str );

    // pack objects the same way
    SimpleObject obj(42);
    o["obj"] = eoserial::pack( obj );

    // pack vector and list the same way
    std::vector<int> vec;
    vec.push_back(1);
    vec.push_back(3);
    vec.push_back(3);
    vec.push_back(7);
    o["vec"] = eoserial::pack( vec );

    std::map<std::string, int> str2int;
    str2int["one"] = 1;
    str2int["two"] = 2;
    str2int["answer"] = 42;
    o["map"] = eoserial::pack( str2int );

    // print it
    o.print( std::cout );

    std::cout << "unpacking..." << std::endl;

    // unpack as easily raw types
    long oneTwoThreeFourFiveSix = 0L;
    eoserial::unpack( o, "long", oneTwoThreeFourFiveSix);
    std::cout << "the long: " << oneTwoThreeFourFiveSix << std::endl;

    // since vec is encoded as an internal eoserial::Array, it can be
    // decoded into a std::vector or a std::list without difference.
    std::list<int> lis;
    eoserial::unpack( o, "vec", lis );

    std::cout << "the list: ";
    for( auto it = lis.begin(), end = lis.end(); it != end; ++it)
    {
        std::cout << *it << ';';
    }
    std::cout << std::endl;

    std::map< std::string, int > readMap;
    eoserial::unpack( o, "map", readMap );
    std::cout << "The answer is " << readMap["answer"] << std::endl;

    obj.value = -1;
    // unpack object the same way
    eoserial::unpack( o, "obj", obj );
    std::cout << "obj.value = " << obj.value << std::endl;

    return 0;
}

@endcode
 *
 * @author Benjamin Bouvier <benjamin.bouvier@gmail.com>
 */

namespace eoserial
{
    /* *****************
     * DESERIALIZATION *
     ******************/

    /**
     * @brief Recursively unpack elements of an object which implements push_back.
     */
    template< class T >
    inline void unpackBasePushBack( const Entity* obj, T& container )
    {
        const Array* arr = static_cast<const Array*>( obj );
        for( Array::const_iterator it = arr->begin(), end = arr->end();
            it != end;
            ++it )
        {
            typename T::value_type item;
            unpackBase( *it, item );
            container.push_back( item );
        }
    }

    /**
     * @brief Unpack method for std::vector
     */
    template< class T >
    inline void unpackBase( const Entity* obj, std::vector<T>& v )
    {
        unpackBasePushBack( obj, v );
    }

    /**
     * @brief Unpack method for std::list
     */
    template< class T >
    inline void unpackBase( const Entity* obj, std::list<T>& l )
    {
        unpackBasePushBack( obj, l );
    }

    /**
     * @brief Unpack method for std::map< std::string, T >
     */
    template< class T >
    inline void unpackBase( const Entity* entity, std::map<std::string, T> & m )
    {
        const Object* obj = static_cast< const Object* >( entity );
        for( Object::const_iterator it = obj->begin(), end = obj->end();
                it != end;
                ++it )
        {
            unpackBase( it->second, m[ it->first ] );
        }
    }

    /**
     * @brief Unpack implementation for non eoserial::Persistent objects.
     *
     * This implementation is being used for every objects that can be transmitted
     * to a std::ostream (i.e. which implements the operator <<)
     */
    template<class T, int n>
    struct UnpackImpl
    {
        void operator()( const Entity* obj, T& value )
        {
            static_cast<const String*>( obj )->deserialize( value );
        }
    };

    /**
     * @brief Unpack implementation for eoserial::Persistent objects.
     */
    template<class T>
    struct UnpackImpl<T, 1>
    {
        void operator()( const Entity* obj, T& value )
        {
            value.unpack( static_cast<const Object*>(obj) );
        }
    };

    /**
     * @brief Unpack helper for determining which implementation to use.
     *
     * The trick comes from Herb Sutter: IsDerivedFrom<T, Persistent>::value is
     * true if and only if T inherits from Persistent. In this case, it's equal
     * to 1, thus the partial specialization of UnpackImpl is used. In the other
     * case, it's equal to 0 and the generic implementation is used.
     */
    template<class T>
    inline void unpackBase( const Entity* obj, T& value )
    {
        UnpackImpl< T, IsDerivedFrom< T, Persistent >::value > impl;
        impl( obj, value );
    }

    /**
     * @brief Universal unpack method.
     *
     * @param obj The eoserial::object containing the value to deserialize.
     * @param key Name of the field to deserialize
     * @param value The object in which we'll store the deserialized value.
     */
    template<class T>
    inline void unpack( const Object& obj, const std::string& key, T& value )
    {
        unpackBase( obj.find(key)->second, value );
    }

    /* *******************
     * SERIALIZATION *****
     ********************/

    /**
     * @brief Pack implementation for non eoserial::Printable objects.
     *
     * This implementation is being used for every objects that can be transmitted
     * to a std::istream (i.e. which implements the operator >>)
     */
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

    /**
     * @brief Pack implementation for eoserial::Printable objects.
     */
    template<class T>
    struct PackImpl<T, 1>
    {
        Entity* operator()( const T& value )
        {
            return value.pack();
        }
    };

    // Pre declaration for being usable in packIterable.
    template<class T>
    inline Entity* pack( const T& value );

    /**
     * @brief Pack method for iterable (begin, end) objects.
     */
    template<class T>
    inline Entity* packIterable( const T& container )
    {
        Array* arr = new Array;
        for( Array::const_iterator it = container.begin(), end = container.end();
                it != end;
                ++it )
        {
            arr->push_back( pack(*it) );
        }
        return arr;
    }

    /**
     * @brief Pack method for std::vector
     */
    template<class T>
    inline Entity* pack( const std::vector<T>& v )
    {
        return packIterable( v );
    }

    /**
     * @brief Pack method for std::list
     */
    template<class T>
    inline Entity* pack( const std::list<T>& l )
    {
        return packIterable( l );
    }

    /**
     * @brief Pack method for std::map< std::string, T >
     */
    template<class T>
    inline Entity* pack( const std::map<std::string, T>& map )
    {
        Object* obj = new Object;
        for( Object::const_iterator it = map.begin(), end = map.end();
                it != end;
                ++it )
        {
            (*obj)[ it->first ] = pack( it->second );
        }
        return obj;
    }

    /**
     * @brief Universal pack method.
     *
     * @see unpackBase to understand the trick with the implementation.
     *
     * @param value Variable to store into an entity.
     * @return An entity to store into an object.
     */
    template<class T>
    inline Entity* pack( const T& value )
    {
        PackImpl<T, IsDerivedFrom< T, Printable >::value> impl;
        return impl( value );
    }

    /** **************
     * OLD FUNCTIONS *
     ****************
    These functions are useful for casting eoserial::objects into simple, primitive
    variables or into class instance which implement eoserial::Persistent.

    The model is always quite the same :
    - the first argument is the containing object (which is a eoserial::Entity, 
    an object or an array)
    - the second argument is the key or index,
    - the last argument is the value in which we're writing.

    */
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
