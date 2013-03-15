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
 * See the snippet example code below.
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
 *  eoserial::Entity* pack() const
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
 * return 0;
 * }
 * @endcode
 *
 * @todo Encapsulate non private functions. As of today (2013-03-19), it is not
 * possible as GCC and Clang refuse to have std::vector specializations of template methods (while it works with
 * functions).
 *
 * @todo Comments coming soon.
 *
 * @author Benjamin Bouvier
 */

# include <vector>
# include <string>
# include <stdexcept>   // std::runtime_error
# include <type_traits> // std::is_convertible (C++11)

# include "SerialString.h"
# include "SerialObject.h"
# include "SerialArray.h"
# include "Utils.h"

namespace eoserial
{

    template<class T>
        eoserial::Entity* serialize( const T & arg );

    template<class T>
        void deserialize( const eoserial::Entity& json, const std::string& field, T & value );

    /* *************************
     * PRIVATE FUNCTIONS *******
     * Use at your own risk! ***
     **************************/

    template<class T>
        eoserial::Entity* makeSimple( const T & arg )
        {
            throw std::runtime_error("eoSerial: makeSimple called with an unknown basic type.");
            return 0;
        }

    template<class T>
        eoserial::Entity* makeObject( const T & arg )
        {
            throw std::runtime_error("eoSerial: makeObject called with an non eoserial::Printable type.");
            return 0;
        }


    template<class T>
        void deserializeSimple( const eoserial::Entity* json, T & value )
        {
            std::runtime_error("eoSerial: deserializeSimple called with an unknown basic type.");
        }

    template<class T>
        void deserializeObject( const eoserial::Entity* json, T & value )
        {
            std::runtime_error("eoSerial:: deserializeObject called with a non eoserial::Persistent object.");
        }

    template<>
        eoserial::Entity* makeObject( const eoserial::Printable & arg )
        {
            return arg.pack();
        }


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
        MKSIMPLE(std::string)
# undef MKSIMPLE


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
    DSSIMPLE(std::string);
# undef DSSIMPLE

    template<>
        void deserializeObject( const eoserial::Entity* json, eoserial::Persistent & value )
        {
            value.unpack( static_cast<const eoserial::Object*>( json ) );
        }

    template<class V>
        eoserial::Entity* makeSimple( const std::vector<V> & v )
        {
            eoserial::Array* array = new eoserial::Array;
            for( auto it = v.begin(), end = v.end();
                    it != end;
                    ++it )
            {
                array->push_back( eoserial::serialize( *it ) );
            }
            return array;
        }

    template<class T>
        void deserializeBase( const eoserial::Entity* json, T & value );

    template< class T  >
        void deserializeSimple( const eoserial::Entity* json, std::vector<T> & v )
        {
            const eoserial::Array* sArray = static_cast<const eoserial::Array*>(json);
            for( auto it = sArray->begin(), end = sArray->end();
                    it != end;
                    ++it )
            {
                T single;
                eoserial::deserializeBase( *it, single );
                v.push_back( single );
            }
        }

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

    template<class T>
        eoserial::Entity* serialize( const T & arg )
        {
            if( std::is_convertible<T*, eoserial::Printable*>::value )
            {
                return eoserial::makeObject( reinterpret_cast<const eoserial::Printable&>(arg) );
            } else {
                return eoserial::makeSimple( arg );
            }
        }

    template<class T>
        void deserialize( const eoserial::Entity& json, const std::string& field, T & value )
        {
            const eoserial::Entity* jField = static_cast<const eoserial::Object&>(json).find(field)->second;
            eoserial::deserializeBase( jField, value );
        }
}


