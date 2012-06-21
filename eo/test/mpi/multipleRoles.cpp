# include <mpi/eompi.h>
# include <mpi/eoParallelApply.h>

# include <boost/serialization/vector.hpp>

# include <iostream>

# include <vector>
using namespace std;

// Role map
// 0 : general master
// 1 : worker of general job, master of subjob
// 2 and more : workers of subjob

struct plusOne : public eoUF< int&, void >
{
    void operator() ( int & x )
    {
        cout << "Subjob is being applied." << endl;
        ++x;
    }
};

void subtask( vector<int>& v )
{
    DynamicAssignmentAlgorithm algo( 2, MpiNode::comm().size()-1 );
    plusOne plusOneInstance;
    ParallelApply<int> job( plusOneInstance, v, algo, 1 );
    Role<int> node( job );
    node.run();
}

struct transmit : public eoUF< vector<int>&, void >
{
    void operator() ( vector<int>& v )
    {
        cout << "Into the master subjob..." << endl;
        subtask( v );
            }
};

int main(int argc, char** argv)
{
    MpiNode::init( argc, argv );
    vector<int> v;

    v.push_back(1);
    v.push_back(3);
    v.push_back(3);
    v.push_back(7);
    v.push_back(42);

    transmit transmitInstance;

    vector< vector<int> > metaV;
    metaV.push_back( v );

    switch( MpiNode::comm().rank() )
    {
        case 0:
        case 1:
            {
                // only one node is assigned to subjob mastering
                DynamicAssignmentAlgorithm algo( 1, 1 );
                ParallelApply< vector<int> > job( transmitInstance, metaV, algo, 0 );
                Role< vector<int> > node( job );
                node.run();
                if( node.master() )
                {
                    v = metaV[0];
                    cout << "Results : " << endl;
                    for(int i = 0; i < v.size(); ++i)
                    {
                        cout << v[i] << ' ';
                    }
                    cout << endl;
                }
            }
            break;

        default:
            {
                // all the other nodes are sub workers
                subtask( v );
            }
            break;
    }

    return 0;
}
