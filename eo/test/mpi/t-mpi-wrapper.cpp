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
 * This file shows an example of how to wrap a handler of a job store. Here, the wrapped handler is the "IsFinished"
 * one. The only function that has been added is that the wrapper prints a message on standard output, indicating what
 * the wrapped function returns as a result.
 *
 * This test is performed on a parallel apply job, the same as in parallelApply. The main difference is when
 * instanciating the store.
 */

# include <mpi/eoMpi.h>
# include <mpi/eoParallelApply.h>
# include <mpi/eoTerminateJob.h>

# include "t-mpi-common.h"

# include <iostream>
# include <cstdlib>

# include <vector>
using namespace std;

using namespace eo::mpi;

// Job functor.
struct plusOne : public eoUF< SerializableBase<int>&, void >
{
    void operator() ( SerializableBase<int>& x )
    {
        ++x;
    }
};

/*
 * Shows the wrapped result of IsFinished, prints a message and returns the wrapped value.
 * times is an integer counting how many time the wrapper (hence the wrapped too) has been called.
 */
template< class EOT >
struct ShowWrappedResult : public IsFinishedParallelApply<EOT>
{
    using IsFinishedParallelApply<EOT>::_wrapped;

    ShowWrappedResult ( IsFinishedParallelApply<EOT> * w = 0 ) : IsFinishedParallelApply<EOT>( w ), times( 0 )
    {
        // empty
    }

    bool operator()()
    {
        bool wrappedValue = _wrapped->operator()(); // (*_wrapped)();
        cout << times << ") Wrapped function would say that it is " << ( wrappedValue ? "":"not ") << "finished" << std::endl;
        ++times;
        return wrappedValue;
    }

    private:
    int times;
};

int main(int argc, char** argv)
{
    // eo::log << eo::setlevel( eo::debug );
    eo::log << eo::setlevel( eo::quiet );

    Node::init( argc, argv );

    srand( time(0) );
    vector< SerializableBase<int> > v;
    for( int i = 0; i < 1000; ++i )
    {
        v.push_back( rand() );
    }

    int offset = 0;
    vector< SerializableBase<int> > originalV = v;

    plusOne plusOneInstance;

    StaticAssignmentAlgorithm assign( v.size() );

    ParallelApplyStore< SerializableBase<int> > store( plusOneInstance, eo::mpi::DEFAULT_MASTER, 1 );
    store.data( v );
    // This is the only thing which changes: we wrap the IsFinished function.
    // According to RAII, we'll delete the invokated wrapper at the end of the main ; the store won't delete it
    // automatically.
    ShowWrappedResult< SerializableBase<int> > wrapper;
    store.wrapIsFinished( &wrapper );

    ParallelApply< SerializableBase<int> > job( assign, eo::mpi::DEFAULT_MASTER, store );
    // Equivalent to:
    // Job< ParallelApplyData<int> > job( assign, 0, store );
    job.run();
    EmptyJob stop( assign, eo::mpi::DEFAULT_MASTER );

    if( job.isMaster() )
    {
        ++offset;
        for(unsigned i = 0; i < v.size(); ++i)
        {
            cout << v[i] << ' ';
            if( originalV[i] + offset != v[i] )
            {
                cout << " <-- ERROR at this point." << endl;
                exit( EXIT_FAILURE );
            }
        }
        cout << endl;
    }

    return 0;
}

