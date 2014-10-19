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
 * This file shows an example of use of parallel apply, in the following context: each element of a table is
 * incremented... in a parallel fashion. While this operation is very easy to perform even on a single host, it's just
 * an example for parallel apply use.
 *
 * The table of integers has to be serialized before it's sent. The wrapper object SerializableBase allows to serialize
 * any type and manipulate it like this type: SerializableBase<int> can be exactly be used as an integer.
 *
 * Besides, this is also a test for assignment (scheduling) algorithms, in different cases. The test succeeds if and
 * only if the program terminates without any segfault ; otherwise, there could be a deadlock which prevents the end or
 * a segfault at any time.
 *
 * One important thing is to instanciate an EmptyJob after having launched a ParallelApplyJob, so as the workers to be
 * aware that the job is done (as it's a MultiJob).
 *
 * This test needs at least 3 processes to be launched. Under this size, it will directly throw an exception, at the
 * beginning;
 */

# include <paradiseo/eompi.h>
# include <paradiseo/eompi/eoParallelApply.h>
# include <paradiseo/eompi/eoTerminateJob.h>

# include "t-mpi-common.h"

# include <iostream>
# include <cstdlib>

# include <vector>
using namespace std;

using namespace eo::mpi;

/*
 * The function to be called on each element of the table: just increment the value.
 */
struct plusOne : public eoUF< SerializableBase<int>&, void >
{
    void operator() ( SerializableBase<int> & x )
    {
        ++x; // implicit conversion of SerializableBase<int> in the integer it contains
    }
};

/*
 * Internal structure representating a test.
 */
struct Test
{
    AssignmentAlgorithm * assign;   // used assignment algorithm for this test.
    string description;             // textual description of the test
    int requiredNodesNumber;        // number of required nodes. NB : chosen nodes ranks must be sequential
};

int main(int argc, char** argv)
{
    // eo::log << eo::setlevel( eo::debug ); // if you like tty full of rainbows, decomment this line and comment the following one.
    eo::log << eo::setlevel( eo::quiet );

    bool launchOnlyOne = false ; // Set this to true if you wanna launch only the first test.

    Node::init( argc, argv );

    // Initializes a vector with random values.
    srand( time(0) );
    vector< SerializableBase<int> > v;
    for( int i = 0; i < 1000; ++i )
    {
        v.push_back( rand() );
    }

    // We need to be sure the values are correctly incremented between each test. So as to check this, we save the
    // original vector into a variable originalV, and put an offset variable to 0. After each test, the offset is
    // incremented and we can compare the returned value of each element to the value of each element in originalV +
    // offset. If the two values are different, there has been a problem.
    int offset = 0;
    vector< SerializableBase<int> > originalV = v;

    // Instanciates the functor to apply on each element
    plusOne plusOneInstance;

    vector< Test > tests;

    const int ALL = Node::comm().size();
    if( ALL < 3 ) {
        throw std::runtime_error("Needs at least 3 processes to be launched!");
    }

    // Tests are auto described thanks to member "description"
    Test tIntervalStatic;
    tIntervalStatic.assign = new StaticAssignmentAlgorithm( 1, REST_OF_THE_WORLD, v.size() );
    tIntervalStatic.description = "Correct static assignment with interval."; // workers have ranks from 1 to size - 1
    tIntervalStatic.requiredNodesNumber = ALL;
    tests.push_back( tIntervalStatic );

    if( !launchOnlyOne )
    {
        Test tWorldStatic;
        tWorldStatic.assign = new StaticAssignmentAlgorithm( v.size() );
        tWorldStatic.description = "Correct static assignment with whole world as workers.";
        tWorldStatic.requiredNodesNumber = ALL;
        tests.push_back( tWorldStatic );

        Test tStaticOverload;
        tStaticOverload.assign = new StaticAssignmentAlgorithm( v.size()+100 );
        tStaticOverload.description = "Static assignment with too many runs.";
        tStaticOverload.requiredNodesNumber = ALL;
        tests.push_back( tStaticOverload );

        Test tUniqueStatic;
        tUniqueStatic.assign = new StaticAssignmentAlgorithm( 1, v.size() );
        tUniqueStatic.description = "Correct static assignment with unique worker.";
        tUniqueStatic.requiredNodesNumber = 2;
        tests.push_back( tUniqueStatic );

        Test tVectorStatic;
        vector<int> workers;
        workers.push_back( 1 );
        workers.push_back( 2 );
        tVectorStatic.assign = new StaticAssignmentAlgorithm( workers, v.size() );
        tVectorStatic.description = "Correct static assignment with precise workers specified.";
        tVectorStatic.requiredNodesNumber = 3;
        tests.push_back( tVectorStatic );

        Test tIntervalDynamic;
        tIntervalDynamic.assign = new DynamicAssignmentAlgorithm( 1, REST_OF_THE_WORLD );
        tIntervalDynamic.description = "Dynamic assignment with interval.";
        tIntervalDynamic.requiredNodesNumber = ALL;
        tests.push_back( tIntervalDynamic );

        Test tUniqueDynamic;
        tUniqueDynamic.assign = new DynamicAssignmentAlgorithm( 1 );
        tUniqueDynamic.description = "Dynamic assignment with unique worker.";
        tUniqueDynamic.requiredNodesNumber = 2;
        tests.push_back( tUniqueDynamic );

        Test tVectorDynamic;
        tVectorDynamic.assign = new DynamicAssignmentAlgorithm( workers );
        tVectorDynamic.description = "Dynamic assignment with precise workers specified.";
        tVectorDynamic.requiredNodesNumber = tVectorStatic.requiredNodesNumber;
        tests.push_back( tVectorDynamic );

        Test tWorldDynamic;
        tWorldDynamic.assign = new DynamicAssignmentAlgorithm;
        tWorldDynamic.description = "Dynamic assignment with whole world as workers.";
        tWorldDynamic.requiredNodesNumber = ALL;
        tests.push_back( tWorldDynamic );
    }

    for( unsigned int i = 0; i < tests.size(); ++i )
    {
        // Instanciates a store with the functor, the master rank and size of packet (see ParallelApplyStore doc).
        ParallelApplyStore< SerializableBase<int> > store( plusOneInstance, eo::mpi::DEFAULT_MASTER, 3 );
        // Updates the contained data
        store.data( v );
        // Creates the job with the assignment algorithm, the master rank and the store
        ParallelApply< SerializableBase<int> > job( *(tests[i].assign), eo::mpi::DEFAULT_MASTER, store );

        // Only master writes information
        if( job.isMaster() )
        {
            cout << "Test : " << tests[i].description << endl;
        }

        // Workers whose rank is inferior to required nodes number have to run the test, the other haven't anything to
        // do.
        if( Node::comm().rank() < tests[i].requiredNodesNumber )
        {
            job.run();
        }

        // After the job run, the master checks the result with offset and originalV
        if( job.isMaster() )
        {
            // This job has to be instanciated, not launched, so as to tell the workers they're done with the parallel
            // job.
            EmptyJob stop( *(tests[i].assign), eo::mpi::DEFAULT_MASTER ); 
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

        // MPI synchronization (all the processes wait to be here).
        Node::comm().barrier();

        delete tests[i].assign;
    }
    return 0;
}

