/*
(c) Benjamin Bouvier, 2013

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

/**
 * @file Serialize.h
 *
 * This file contains primitive to make serialization and
 * deserialization easy.
 *
 * See the example code snippet below.
 *
 * @code
 * # include <eoSerial.h>
 * # include <string>
 *
 * class MyObject: public eoserial::Persistent
 * {
 *  public:
 *
 *  int value;
 *  MyObject( int v ) : value(v) {}
 *
 *  eoserial::Object* pack() const
 *  {
 *      eoserial::Object* e = new eoserial::Object;
 *      (*e)["the_value"] = eoserial::serialize( value );
 *      return e;
 *  }
 *
 *  void unpack( const eoserial::Object* o )
 *  {
 *      eoserial::deserialize( *o, "the_value", value );
 *  }
 * };
 *
 * int main()
 * {
 * eoserial::Object o;
 * o["long"] = eoserial::serialize(123456L);
 * o["bool"] = eoserial::serialize(true);
 * o["double"] = eoserial::serialize(3.141592653);
 * o["float"] = eoserial::serialize(3.141592653f);
 *
 * std::string str = "Hello, world!";
 * o["str"] = eoserial::serialize( str );
 *
 * MyObject obj(42);
 * o["obj"] = eoserial::serialize( obj );
 *
 * std::vector<int> vec;
 * vec.push_back(1);
 * vec.push_back(3);
 * vec.push_back(3);
 * vec.push_back(7);
 * o["vec"] = eoserial::serialize( vec );
 *
 * o.print( std::cout );
 *
 * std::list<int> lis;
 * eoserial::deserialize( o, "vec", lis );
 *
 * long oneTwoThreeFourFiveSix;
 * eoserial::deserialize( o, "long", oneTwoThreeFourFiveSix);
 *
 * return 0;
 * }
 * @endcode
 *
 * @todo Encapsulate private functions. As of today (2013-03-19), it is not
 * possible as GCC and Clang refuse to have std::vector specializations of template methods (while it works with
 * functions).
 *
 * @todo Comments coming soon.
 *
 * @author Benjamin Bouvier
 */

# include <vector>
# include <list>
# include <string>
# include <stdexcept>   // std::runtime_error
# include <type_traits> // std::is_convertible (C++11)

# include "SerialString.h"
# include "SerialObject.h"
# include "SerialArray.h"
# include "Utils.h"

namespace eoserial
{

    /**
     * @brief Tries to serialize the given argument into an Entity.
     *
     * This function can be called with any argument of the following kinds:
     * - basic types (int, bool, float, double, char, short, long)
     * - standard STL types (std::string, std::list, std::vector). In
     *   this case, the serialization is automatically called on the
     *   contained objects.
     * - objects which implement eoserial::Printable
     *
     * @param arg The argument to serialize.
     * @return an Entity to be used with the serialization module.
     *
     * @throws std::runtime_exception when the type T is not known or
     * not convertible to a known type.
     */
    template<class T>
        eoserial::Entity* serialize( const T & arg );

    /**
     * @brief Tries to deserialize the given argument from the given field in the entity and loads it into the in-out
     * given value.
     *
     * This function is the reverse operator of the serialize function:
     * - basic types are supported
     * - standard STL types (std::string, std::list, std::vector)
     * - objects which implement eoserial::Persistent
     * @see serialize
     *
     * @param json The entity containing the variable to deserialize
     * @param field The name of the field used in the original object
     * @param value The in-out value in which we want to store the result of the deserialization.
     *
     * @throws std::runtime_exception when the type T is not known or
     * not convertible to a known type.
     */
    template<class T>
        void deserialize( const eoserial::Entity& json, const std::string& field, T & value );

    /* *************************
     * PRIVATE FUNCTIONS *******
     * Use at your own risk! ***
     **************************/

    /* *************************
     * SERIALIZATION ***********
     **************************/

    /**
     * @brief Function to be called for non eoserial::Printable objects.
     *
     * The default behaviour is to throw a runtime exception. The function is then specialized for known types.
     */
    template<class T>
        eoserial::Entity* makeSimple( const T & arg )
        {
            throw std::runtime_error("eoSerial: makeSimple called with an unknown basic type.");
            return 0;
        }

    /**
     * @brief Function to be called for eoserial::Printable objects and only these ones.
     *
     * The default behaviour is to throw a runtime exception. The function is specialized only for eoserial::Printable
     * object.
     */
    template<class T>
        eoserial::Entity* makeObject( const T & arg )
        {
            throw std::runtime_error("eoSerial: makeObject called with an non eoserial::Printable type.");
            return 0;
        }

    /*
     * Specializations of makeSimple<T> for basic types.
     * Macro MKSIMPLE can be used to register any type that can be printed into a std::ostream.
     */
# define MKSIMPLE(A) template<>\
    eoserial::Entity* makeSimple( const A& arg ) \
    { \
        return eoserial::make(arg); \
    } \

    MKSIMPLE(bool)
        MKSIMPLE(int)
        MKSIMPLE(short)
        MKSIMPLE(long)
        MKSIMPLE(float)
        MKSIMPLE(double)
        MKSIMPLE(char)
        MKSIMPLE(std::string)
# undef MKSIMPLE // avoids undebuggable surprises

        /**
         * @brief Base specialization for objects iterable thanks to
         * begin(), end() and basic iterators.
         *
         * This specialization is used for std::list and std::vector.
         */
    template<class Container>
        eoserial::Entity* makeSimpleIterable( const Container & c )
        {
            eoserial::Array* array = new eoserial::Array;
            for( auto it = c.begin(), end = c.end();
                    it != end;
                    ++it )
            {
                array->push_back( eoserial::serialize( *it ) );
            }
            return array;
        }

    template<class V>
        eoserial::Entity* makeSimple( const std::vector<V> & v )
        {
            return makeSimpleIterable( v );
        }

    template<class V>
        eoserial::Entity* makeSimple( const std::list<V> & l )
        {
            return makeSimpleIterable( l );
        }

    /**
     * @brief Specialization of makeObject for eoserial::Printable.
     *
     * For these objects, we can directly use their pack method.
     */
    template<>
        eoserial::Entity* makeObject( const eoserial::Printable & arg )
        {
            return arg.pack();
        }

    /*
     * @brief Implementation of Serialize function.
     *
     * The idea is the following:
     * - either the object implements eoserial::Printable and can be serialized directly with makeObject
     * - or it's not, and thus we have to try the registered types.
     *
     * The difficulty of this function is to be callable with any kind of argument, whatever the type. For that purpose,
     * templating is frequently used with default behaviours being erroneous. This way, the compiler can try all
     * branches of the conditional and find an implementation of the function that works for the given type.
     * This trick is used as the templates functions (and methods) are invariant in C++: if A inherits B, the specialization
     * f<B>() is not used when calling with the parameter A.
     */
    template<class T>
        eoserial::Entity* serialize( const T & arg )
        {
            // static check (introduced by C++11)
            // - avoids the invariant template issue
            if( std::is_convertible<T*, eoserial::Printable*>::value )
            {
                // at this point, we are sure that we can cast the argument into an eoserial::Printable.
                // reinterpret_cast has to be used, otherwise static_cast and dynamic_cast will fail at compile time for
                // basic types.
                return eoserial::makeObject( reinterpret_cast<const eoserial::Printable&>(arg) );
            } else {
                // not an eoserial::Printable, try registered types
                return eoserial::makeSimple( arg );
            }
        }


    /* *****************
     * DESERIALIZATION *
     ******************/

    /**
     * @brief Function to be called for non eoserial::Persistent objects.
     *
     * The default behaviour is to throw a runtime exception. The function is then specialized for known types.
     */
    template<class T>
        void deserializeSimple( const eoserial::Entity* json, T & value )
        {
            throw std::runtime_error("eoSerial: deserializeSimple called with an unknown basic type.");
        }

    /**
     * @brief Function to be called for eoserial::Persistent objects and only these ones.
     *
     * The default behaviour is to throw a runtime exception. The function is specialized only for eoserial::Persistent
     * object.
     */
    template<class T>
        void deserializeObject( const eoserial::Entity* json, T & value )
        {
            throw std::runtime_error("eoSerial:: deserializeObject called with a non eoserial::Persistent object.");
        }

    /*
     * Specializations of deserializeSimple<T> for basic types.
     * Macro DSSIMPLE can be used to register any type that can be read from a std::istream.
     */
# define DSSIMPLE(A) template<> \
        void deserializeSimple( const eoserial::Entity* json, A & value ) \
        { \
            static_cast<const eoserial::String*>(json)->deserialize( value ); \
        }

        DSSIMPLE(bool);
    DSSIMPLE(int);
    DSSIMPLE(short);
    DSSIMPLE(long);
    DSSIMPLE(float);
    DSSIMPLE(double);
    DSSIMPLE(char);
    DSSIMPLE(std::string);
# undef DSSIMPLE // avoids undebuggable surprises

    /**
     * @brief Deserialize function with two arguments.
     *
     * Used by list and vector containers.
     */
    template<class T>
        void deserializeBase( const eoserial::Entity* json, T & value );

    /**
     * @brief Base specialization for objects that implement push_back.
     *
     * This specialization is used for std::list and std::vector.
     */
    template< class Container >
        void deserializeSimplePushBack( const eoserial::Entity* json, Container & c )
        {
            const eoserial::Array* sArray = static_cast<const eoserial::Array*>(json);
            for( auto it = sArray->begin(), end = sArray->end();
                    it != end;
                    ++it )
            {
                typename Container::value_type single;
                eoserial::deserializeBase( *it, single );
                c.push_back( single );
            }
        }

    template< class T >
        void deserializeSimple( const eoserial::Entity* json, std::vector<T> & v )
        {
            deserializeSimplePushBack( json, v );
        }

    template< class T >
        void deserializeSimple( const eoserial::Entity* json, std::list<T> & v )
        {
            deserializeSimplePushBack( json, v );
        }

    /**
     * @brief Specialization of deserializeObject for eoserial::Persistent.
     *
     * For these objects, we can directly use their unpack method.
     */
    template<>
        void deserializeObject( const eoserial::Entity* json, eoserial::Persistent & value )
        {
            value.unpack( static_cast<const eoserial::Object*>( json ) );
        }

    /*
     * Implementation of deserializeBase.
     *
     * For technical comments, @see makeSimple. The followed scheme
     * is exactly the same.
     */
    template<class T>
        void deserializeBase( const eoserial::Entity* json, T & value )
        {
            if( std::is_convertible< T*, eoserial::Persistent*>::value )
            {
                eoserial::deserializeObject( json, reinterpret_cast<eoserial::Persistent&>( value ) );
            } else {
                eoserial::deserializeSimple( json, value );
            }
        }

    /*
     * Implementation of deserialize.
     *
     * Simply calls the deserializeBase function with the corresponding Entity.
     */
    template<class T>
        void deserialize( const eoserial::Entity& json, const std::string& field, T & value )
        {
            const eoserial::Entity* jField = static_cast<const eoserial::Object&>(json).find(field)->second;
            eoserial::deserializeBase( jField, value );
        }
}


