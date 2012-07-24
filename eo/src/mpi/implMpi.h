# ifndef __EO_MPI_HPP__
# define __EO_MPI_HPP__

# include <mpi.h>
# include <serial/eoSerial.h>

namespace mpi
{
    extern const int any_source;
    extern const int any_tag;

    class environment
    {
        public:

        environment(int argc, char**argv);

        ~environment();
    };

    class status
    {
        public:

        status( const MPI_Status & s );

        int tag() { return _tag; }
        int error() { return _error; }
        int source() { return _source; }

        private:
            int _source;
            int _tag;
            int _error;
    };

    class communicator
    {
        public:

        communicator( );

        ~communicator();

        int rank();

        int size();

        /*
         * SEND / RECV INT
         */
        void send( int dest, int tag, int n );

        void recv( int src, int tag, int& n );

        /*
         * SEND / RECV STRING
         */
        void send( int dest, int tag, const std::string& str );

        void recv( int src, int tag, std::string& str );

        /*
         * SEND / RECV Objects
         */
        void send( int dest, int tag, const eoserial::Persistent & persistent );

        template< class T >
        void send( int dest, int tag, T* table, int size )
        {
            // Puts all the values into an array
            eoserial::Array* array = new eoserial::Array;

            for( int i = 0; i < size; ++i )
            {
                array->push_back( table[i].pack() );
            }

            // Encapsulates the array into an object
            eoserial::Object* obj = new eoserial::Object;
            obj->add( "array", array );
            std::stringstream ss;
            obj->print( ss );
            delete obj;

            // Sends the object as a string
            send( dest, tag, ss.str() );
        }

        void recv( int src, int tag, eoserial::Persistent & persistent );

        template< class T >
        void recv( int src, int tag, T* table, int size )
        {
            // Receives the string which contains the object
            std::string asText;
            recv( src, tag, asText );

            // Parses the object and retrieves the table
            eoserial::Object* obj = eoserial::Parser::parse( asText );
            eoserial::Array* array = static_cast<eoserial::Array*>( (*obj)["array"] );

            // Retrieves all the values from the array
            for( int i = 0; i < size; ++i )
            {
                eoserial::unpackObject( *array, i, table[i] );
            }
            delete obj;
        }

        /*
         * Other methods
         */
        status probe( int src = any_source, int tag = any_tag );

        void barrier();

        private:
            int _rank;
            int _size;

            char* _buf;
            int _bufsize;
    };

    void broadcast( communicator & comm, int value, int root );
} // namespace mpi

# endif //__EO_MPI_HPP__
