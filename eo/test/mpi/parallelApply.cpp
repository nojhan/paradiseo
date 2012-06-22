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
};

int main(int argc, char** argv)
{
    // eo::log << eo::setlevel( eo::debug );
    MpiNode::init( argc, argv );

    vector<int> v;
    v.push_back(1);
    v.push_back(3);
    v.push_back(3);
    v.push_back(7);
    v.push_back(42);

    plusOne plusOneInstance;

    vector< Test > tests;

    Test tStatic;
    tStatic.assign = new StaticAssignmentAlgorithm( 1, MpiNode::comm().size()-1, v.size() );
    tStatic.description = "Correct static assignment.";
    tests.push_back( tStatic );

    Test tStaticOverload;
    tStaticOverload.assign = new StaticAssignmentAlgorithm( 1, MpiNode::comm().size()-1, v.size()+100 );
    tStaticOverload.description = "Static assignment with too many runs.";
    tests.push_back( tStaticOverload );

    Test tDynamic;
    tDynamic.assign = new DynamicAssignmentAlgorithm( 1, MpiNode::comm().size()-1 );
    tDynamic.description = "Dynamic assignment.";
    tests.push_back( tDynamic );

    for( unsigned int i = 0; i < tests.size(); ++i )
    {
        ParallelApply<int> job( plusOneInstance, v, *(tests[i].assign), 0 );

        if( job.isMaster() )
        {
            cout << "Test : " << tests[i].description << endl;
        }

        job.run();

        if( job.isMaster() )
        {
            for(int i = 0; i < v.size(); ++i)
            {
                cout << v[i] << ' ';
            }
            cout << endl;
        }

        delete tests[i].assign;
    }
    return 0;
}
