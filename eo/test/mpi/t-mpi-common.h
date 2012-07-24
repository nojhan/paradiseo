# ifndef __T_MPI_COMMON_H__
# define __T_MPI_COMMON_H__

#include <serial/eoSerial.h>

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
