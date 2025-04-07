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
#include "implMpi.h"

namespace eo
{
namespace mpi
{
    const int any_source = MPI_ANY_SOURCE;
    const int any_tag = MPI_ANY_TAG;

    environment::environment(int argc, char**argv)
    {
        MPI_Init(&argc, &argv);
    }

    environment::~environment()
    {
        MPI_Finalize();
    }

    status::status( const MPI_Status & s )
    {
        _source = s.MPI_SOURCE;
        _tag = s.MPI_TAG;
        _error = s.MPI_ERROR;
    }

    communicator::communicator( )
    {
        _rank = -1;
        _size = -1;

        _buf = 0;
        _bufsize = -1;
    }

    communicator::~communicator()
    {
        if( _buf )
        {
            delete _buf;
            _buf = 0;
        }
    }

    int communicator::rank()
    {
        if ( _rank == -1 )
        {
            MPI_Comm_rank( MPI_COMM_WORLD, &_rank );
        }
        return _rank;
    }

    int communicator::size()
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
    void communicator::send( int dest, int tag, int n )
    {
        MPI_Send( &n, 1, MPI_INT, dest, tag, MPI_COMM_WORLD );
    }

    void communicator::recv( int src, int tag, int& n )
    {
        MPI_Status stat;
        MPI_Recv( &n, 1, MPI_INT, src, tag, MPI_COMM_WORLD , &stat );
    }

    /*
     * SEND / RECV STRING
     */
    void communicator::send( int dest, int tag, const std::string& str )
    {
        int size = str.size() + 1;
        send( dest, tag, size );
        MPI_Send( (char*)str.c_str(), size, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
    }

    void communicator::recv( int src, int tag, std::string& str )
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
    void communicator::send( int dest, int tag, const eoserial::Persistent & persistent )
    {
        eoserial::Object* obj = persistent.pack();
        std::stringstream ss;
        obj->print( ss );
        delete obj;
        send( dest, tag, ss.str() );
    }

    void communicator::recv( int src, int tag, eoserial::Persistent & persistent )
    {
        std::string asText;
        recv( src, tag, asText );
        eoserial::Object* obj = eoserial::Parser::parse( asText );
        persistent.unpack( obj );
        delete obj;
    }

    /*
     * Other methods
     */
    status communicator::probe( int src, int tag )
    {
        MPI_Status stat;
        MPI_Probe( src, tag, MPI_COMM_WORLD , &stat );
        return status( stat );
    }

    void communicator::barrier()
    {
        MPI_Barrier( MPI_COMM_WORLD );
    }

    void broadcast( communicator & /*comm*/, int value, int root )
    {
        MPI_Bcast( &value, 1, MPI_INT, root, MPI_COMM_WORLD );
    }
} // namespace mpi
} // namespace eo
