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

/*
 * This file shows an example of how to make a hierarchy between nodes, when using a parallel apply. In this basic
 * test, the master delegates the charge of finding workers to 2 "sub" masters, which then send part of the table to
 * their workers.
 *
 * It's convenient to establish a role map, so as to clearly identify every role:
 * - The node 0 is the general master, that delegates the job. It sends the table to the 2 submasters, and waits for the
 *   results.
 * - Nodes 1 and 2 are the worker of the first job: the delegates. They receive the elements of the table and
 *   retransmit them to the subworkers. They play the roles of worker in the delegating job, and master in the plus one
 *   job.
 * - Following nodes (3 to 6) are workers of the plus one job. They do the real job. Nodes 3 and 5 are attached to
 *   submaster 1, 4 and 6 to submaster 2.
 *
 * This test requires exactly 7 hosts. If the size is bigger, an exception will be thrown at the beginning.
 **/

# include <paradiseo/eompi.h>
# include <paradiseo/eompi/eoParallelApply.h>
# include <paradiseo/eompi/eoTerminateJob.h>

# include "t-mpi-common.h"

# include <iostream>

# include <vector>
using namespace std;

using namespace eo::mpi;

/*
 * This class allows the user to easily serialize a vector of elements which implement eoserial::Persistent too.
 *
 * T is the type of the contained element, which must implement eoserial::Persistent too.
 *
 * Here, it contains SerializableBase<int>, which is a serializable integer that can be used as an integer.
 */
template< class T >
struct SerializableVector : public std::vector<T>, public eoserial::Persistent
{
    public:

    void unpack( const eoserial::Object* obj )
    {
        this->clear();
        eoserial::Array* vector = static_cast<eoserial::Array*>( obj->find("vector")->second );
        vector->deserialize< std::vector<T>, eoserial::Array::UnpackObjectAlgorithm >( *this );
    }

    eoserial::Object* pack( void ) const
    {
        eoserial::Object* obj = new eoserial::Object;
        obj->add("vector", eoserial::makeArray< std::vector<T>, eoserial::SerializablePushAlgorithm >( *this ) );
        return obj;
    }
};

// The real job to execute, for the subworkers: add one to each element of a table.
struct SubWork: public eoUF< SerializableBase<int>&, void >
{
    void operator() ( SerializableBase<int> & x )
    {
        cout << "Subwork phase." << endl;
        ++x;
    }
};

// Function called by both subworkers and delegates.
// v is the vector to process, rank is the MPI rank of the sub master
void subtask( vector< SerializableBase<int> >& v, int rank )
{
    // Attach workers according to nodes.
    // Submaster with rank 1 will have ranks 3 and 5 as subworkers.
    // Submaster with rank 2 will have ranks 4 and 6 as subworkers.
    vector<int> workers;
    workers.push_back( rank + 2 );
    workers.push_back( rank + 4 );
    DynamicAssignmentAlgorithm algo( workers );
    SubWork sw;

    // Launch the job!
    ParallelApplyStore< SerializableBase<int> > store( sw, rank );
    store.data( v );
    ParallelApply< SerializableBase<int> > job( algo, rank, store );
    job.run();
    EmptyJob stop( algo, rank );
}

// Functor applied by submasters. Wait for the subworkers responses and then add some random processing (here, multiply
// each result by two).
// Note that this work receives a vector of integers as an entry, while subworkers task's operator receives a simple
// integer.
struct Work: public eoUF< SerializableVector< SerializableBase<int> >&, void >
{
    void operator() ( SerializableVector< SerializableBase<int> >& v )
    {
        cout << "Work phase..." << endl;
        subtask( v, Node::comm().rank() );
        for( unsigned i = 0; i < v.size(); ++i )
        {
            v[i] *= 2;
        }
    }
};

int main(int argc, char** argv)
{
    // eo::log << eo::setlevel( eo::debug );
    Node::init( argc, argv );
    if( Node::comm().size() != 7 ) {
        throw std::runtime_error("World size should be 7.");
    }

    SerializableVector< SerializableBase<int> > v;

    v.push_back(1);
    v.push_back(3);
    v.push_back(3);
    v.push_back(7);
    v.push_back(42);

    // As submasters' operator receives a vector<int> as an input, and ParallelApply takes a vector of
    // operator's input as an input, we have to deal with a vector of vector of integers for the master task.
    vector< SerializableVector< SerializableBase<int> > > metaV;
    // Here, we send twice the same vector. We could also have splitted the first vector into two vectors, one
    // containing the beginning and another one containing the end.
    metaV.push_back( v );
    metaV.push_back( v );

    // Assigning roles is done by comparing MPI ranks.
    switch( Node::comm().rank() )
    {
        // Nodes from 0 to 2 are implicated into the delegating task.
        case 0:
        case 1:
        case 2:
            {
                Work w;
                DynamicAssignmentAlgorithm algo( 1, 2 );
                ParallelApplyStore< SerializableVector< SerializableBase<int> > > store( w, 0 );
                store.data( metaV );
                ParallelApply< SerializableVector< SerializableBase<int> > > job( algo, 0, store );
                job.run();
                if( job.isMaster() )
                {
                    EmptyJob stop( algo, 0 );
                    v = metaV[0];
                    cout << "Results : " << endl;
                    for(unsigned i = 0; i < v.size(); ++i)
                    {
                        cout << v[i] << ' ';
                    }
                    cout << endl;
                }
            }
            break;

        // Other nodes are implicated into the subwork task.
        default:
            {
                // all the other nodes are sub workers
                int rank = Node::comm().rank();
                if ( rank == 3 or rank == 5 )
                {
                    subtask( v, 1 );
                } else {
                    subtask( v, 2 );
                }
            }
            break;
    }

    return 0;
}
