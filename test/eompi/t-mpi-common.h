# ifndef __T_MPI_COMMON_H__
# define __T_MPI_COMMON_H__

#include <paradiseo/eoserial.h>

/**
 * @file t-mpi-common.h
 *
 * This file shows an example of serialization of a primitive type, so as to be used in a parallel algorithm.
 * It is fully compatible with the basic type, by implementing conversion operator and constructor based on type.
 * It can contain any simple type which can be written in a std::ostream.
 */

template< class T >
struct SerializableBase : public eoserial::Persistent
{
    public:

        operator T&()
        {
            return _value;
        }

        SerializableBase() : _value()
        {
            // empty
        }

        SerializableBase( T base ) : _value( base )
        {
            // empty
        }

        void unpack( const eoserial::Object* obj )
        {
            eoserial::unpack( *obj, "value", _value );
        }

        eoserial::Object* pack(void) const
        {
            eoserial::Object* obj = new eoserial::Object;
            obj->add("value", eoserial::make( _value ) );
            return obj;
        }

    private:
        T _value;
};


# endif // __T_MPI_COMMON_H__
