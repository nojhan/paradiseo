# include <mpi/eompi.h>
# include <mpi/eoParallelApply.h>

# include <iostream>

# include <vector>
using namespace std;

struct plusOne : public eoUF< int&, void >
{
    void operator() ( int & x )
    {
        ++x;
    }
};

struct Test
{
    AssignmentAlgorithm * assign;
    string description;
    int requiredNodesNumber; // nb : chosen nodes ranks must be sequential
};

// These tests require at least 3 processes to be launched.
int main(int argc, char** argv)
{
    // eo::log << eo::setlevel( eo::debug );
    bool launchOnlyOne = false; // Set this to true if you wanna launch only the first test.

    MpiNode::init( argc, argv );

    srand( time(0) );
    vector<int> v;
    for( int i = 0; i < 1000; ++i )
    {
        v.push_back( rand() );
    }
    
    int offset = 0;
    vector<int> originalV = v;

    plusOne plusOneInstance;

    vector< Test > tests;
    
    const int ALL = MpiNode::comm().size();

    Test tIntervalStatic;
    tIntervalStatic.assign = new StaticAssignmentAlgorithm( 1, eo::REST_OF_THE_WORLD, v.size() );
    tIntervalStatic.description = "Correct static assignment with interval.";
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
        tIntervalDynamic.assign = new DynamicAssignmentAlgorithm( 1, eo::REST_OF_THE_WORLD );
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
        ParallelApply<int> job( plusOneInstance, v, *(tests[i].assign), 0, 3 );

        if( job.isMaster() )
        {
            cout << "Test : " << tests[i].description << endl;
        }

        if( MpiNode::comm().rank() < tests[i].requiredNodesNumber )
        {
            job.run();
        }

        if( job.isMaster() )
        {
            ++offset;
            for(int i = 0; i < v.size(); ++i)
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

        MpiNode::comm().barrier();

        delete tests[i].assign;
    }
    return 0;
}

