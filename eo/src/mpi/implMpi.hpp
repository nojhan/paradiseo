# ifndef __EO_MPI_HPP__
# define __EO_MPI_HPP__

# include <mpi.h>
# include <serial/eoSerial.h>

# include <fstream>

namespace mpi
{
    const int any_source = MPI_ANY_SOURCE;
    const int any_tag = MPI_ANY_TAG;

    class environment
    {
        public:

        environment(int argc, char**argv)
        {
            MPI_Init(&argc, &argv);
        }

        ~environment()
        {
            MPI_Finalize();
        }
    };

    class status
    {
        public:

        status( const MPI_Status & s )
        {
            _source = s.MPI_SOURCE;
            _tag = s.MPI_TAG;
            _error = s.MPI_ERROR;
        }

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

        communicator( )
        {
            _rank = -1;
            _size = -1;

            _buf = 0;
            _bufsize = -1;
        }

        ~communicator()
        {
            if( _buf )
            {
                delete _buf;
                _buf = 0;
            }
        }

        int rank()
        {
            if ( _rank == -1 )
            {
                MPI_Comm_rank( MPI_COMM_WORLD, &_rank );
            }
            return _rank;
        }

        int size()
        {
            if ( _size == -1 )
            {
                MPI_Comm_size( MPI_COMM_WORLD, &_size );
            }
            return _size;
        }

        /*
         * SEND / RECV INT
         */
        void send( int dest, int tag, int n )
        {
            //send( dest, tag, &n, 1 );
            MPI_Send( &n, 1, MPI_INT, dest, tag, MPI_COMM_WORLD );
        }

        void recv( int src, int tag, int& n )
        {
            MPI_Status stat;
            MPI_Recv( &n, 1, MPI_INT, src, tag, MPI_COMM_WORLD , &stat );
            //recv( src, tag, &n, 1 );
        }

        /*
        void send( int dest, int tag, int* table, int size )
        {
            MPI_Send( table, size, MPI_INT, dest, tag, MPI_COMM_WORLD );
        }

        void recv( int src, int tag, int* table, int size )
        {
            MPI_Status stat;
            MPI_Recv( table, size, MPI_INT, src, tag, MPI_COMM_WORLD , &stat );
        }
        */

        /*
         * SEND / RECV STRING
         */
        void send( int dest, int tag, const std::string& str )
        {
            int size = str.size() + 1;
            send( dest, tag, size );
            MPI_Send( (char*)str.c_str(), size, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
        }

        void recv( int src, int tag, std::string& str )
        {
            int size = -1;
            MPI_Status stat;
            recv( src, tag, size );

            if( _buf == 0 )
            {
                _buf = new char[ size ];
                _bufsize = size;
            } else if( _bufsize < size )
            {
                delete [] _buf;
                _buf = new char[ size ];
                _bufsize = size;
            }
            MPI_Recv( _buf, size, MPI_CHAR, src, tag, MPI_COMM_WORLD, &stat );
            str.assign( _buf );
        }

        /*
         * SEND / RECV Objects
         */
        void send( int dest, int tag, const eoserial::Persistent & persistent )
        {
            eoserial::Object* obj = persistent.pack();
            std::stringstream ss;
            obj->print( ss );
            delete obj;
            send( dest, tag, ss.str() );
        }

        /*
        void send( int dest, int tag, eoserial::Persistent* table, int size )
        {
            // Puts all the values into an array
            eoserial::Array* array = new eoserial::Array;

            std::cout << "DEBUG EO: @ premier: " << table << " / @ second: " << table+1 << std::endl;

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
        */

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


        void recv( int src, int tag, eoserial::Persistent & persistent )
        {
            std::string asText;
            recv( src, tag, asText );
            eoserial::Object* obj = eoserial::Parser::parse( asText );
            persistent.unpack( obj );
            delete obj;
        }

        /*
        void recv( int src, int tag, eoserial::Persistent* table, int size )
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
        */

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
        status probe( int src = any_source, int tag = any_tag )
        {
            MPI_Status stat;
            MPI_Probe( src, tag, MPI_COMM_WORLD , &stat );
            return status( stat );
        }

        void barrier()
        {
            MPI_Barrier( MPI_COMM_WORLD );
        }

        private:
            int _rank;
            int _size;

            char* _buf;
            int _bufsize;
    };

    void broadcast( communicator & comm, int value, int root );
} // namespace mpi

# endif //__EO_MPI_HPP__
