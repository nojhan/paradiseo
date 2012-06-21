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

int main(int argc, char** argv)
{
    cout << "Appel à init... " << endl;
    MpiNode::init( argc, argv );
    DynamicAssignmentAlgorithm algo( 1, MpiNode::comm().size()-1 );

    cout << "Création des données... " << endl;
    vector<int> v;

    v.push_back(1);
    v.push_back(3);
    v.push_back(3);
    v.push_back(7);
    v.push_back(42);

    plusOne plusOneInstance;

    cout << "Création du job..." << endl;
    ParallelApply<int> job( plusOneInstance, v, algo, 0 );
    Role<int> node( job );
    node.run();

    if( node.master() )
    {
        for(int i = 0; i < v.size(); ++i)
        {
            cout << v[i] << ' ';
        }
        cout << endl;
    }

    return 0;
}
