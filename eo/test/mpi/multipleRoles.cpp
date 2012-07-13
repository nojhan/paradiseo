# include <mpi/eoMpi.h>
# include <mpi/eoParallelApply.h>
# include <mpi/eoTerminateJob.h>

# include <boost/serialization/vector.hpp>

# include <iostream>

# include <vector>
using namespace std;

using namespace eo::mpi;

// Role map
// 0 : general master
// 1, 2 : worker of general job, master of subjob
// 3 to 7 : workers of subjob

struct SubWork: public eoUF< int&, void >
{
    void operator() ( int & x )
    {
        cout << "Subwork phase." << endl;
        ++x;
    }
};

void subtask( vector<int>& v, int rank )
{
    vector<int> workers;
    workers.push_back( rank + 2 );
    workers.push_back( rank + 4 );
    DynamicAssignmentAlgorithm algo( workers );
    SubWork sw;

    ParallelApplyStore<int> store( sw, rank );
    store.data( v );
    ParallelApply<int> job( algo, rank, store );
    job.run();
    EmptyJob stop( algo, rank );
}

struct Work: public eoUF< vector<int>&, void >
{
    void operator() ( vector<int>& v )
    {
        cout << "Work phase..." << endl;
        subtask( v, Node::comm().rank() );
        for( int i = 0; i < v.size(); ++i )
        {
            v[i] *= 2;
        }
    }
};

int main(int argc, char** argv)
{
    // eo::log << eo::setlevel( eo::debug );
    Node::init( argc, argv );
    vector<int> v;

    v.push_back(1);
    v.push_back(3);
    v.push_back(3);
    v.push_back(7);
    v.push_back(42);

    vector< vector<int> > metaV;
    metaV.push_back( v );
    metaV.push_back( v );

    switch( Node::comm().rank() )
    {
        case 0:
        case 1:
        case 2:
            {
                Work w;
                DynamicAssignmentAlgorithm algo( 1, 2 );
                ParallelApplyStore< vector<int> > store( w, 0 );
                store.data( metaV );
                ParallelApply< vector<int> > job( algo, 0, store );
                job.run();
                if( job.isMaster() )
                {
                    EmptyJob stop( algo, 0 );
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