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
    /*
    cout << "Création de l'instance..." << endl;
    MpiNode* instance = MpiNodeStore::instance();
    if( dynamic_cast<MasterNode*>( instance ) != 0 )
    {
        cout << "[Master] Algorithme d'assignation" << endl;
        static_cast<MasterNode*>( instance )->setAssignmentAlgorithm( &algo );
        cout << "[Master] Lancement." << endl;
        static_cast<MasterNode*>( instance )->run( job );

        for (int i = 0; i < v.size(); ++i )
        {
            cout << v[i] << endl;
        }
    } else if ( dynamic_cast<WorkerNode*>( instance ) != 0 )
    {
        cout << "[Worker] Lancement." << endl;
        static_cast<WorkerNode*>( instance )->run( job );
    } else
    {
        cout << "Nothing to be done;" << endl;
    }
    */

    return 0;
}
