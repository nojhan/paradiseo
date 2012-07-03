# include <mpi/eoMpi.h>
# include <mpi/eoParallelApply.h>

# include <iostream>

# include <vector>
using namespace std;

using namespace eo::mpi;

struct plusOne : public eoUF< int&, void >
{
    void operator() ( int & x )
    {
        ++x;
    }
};

template< class EOT >
struct ShowWrappedResult : public IsFinishedParallelApply<EOT>
{
    using IsFinishedParallelApply<EOT>::wrapped;

    ShowWrappedResult ( IsFinishedParallelApply<EOT> * w ) : IsFinishedParallelApply<EOT>( w ), times( 0 )
    {
        // empty
    }

    bool operator()()
    {
        bool wrappedValue = wrapped->operator()(); // (*wrapped)();
        cout << times << ") Wrapped function would say that it is " << ( wrappedValue ? "":"not ") << "finished" << std::endl;
        ++times;
        return wrappedValue;
    }

    private:
    int times;
};

// These tests require at least 3 processes to be launched.
int main(int argc, char** argv)
{
    // eo::log << eo::setlevel( eo::debug );
    eo::log << eo::setlevel( eo::quiet );

    Node::init( argc, argv );

    srand( time(0) );
    vector<int> v;
    for( int i = 0; i < 1000; ++i )
    {
        v.push_back( rand() );
    }

    int offset = 0;
    vector<int> originalV = v;

    plusOne plusOneInstance;

    StaticAssignmentAlgorithm assign( v.size() );

    ParallelApplyStore< int > store( plusOneInstance, v, 0, 1 );
    IsFinishedParallelApply< int >& wrapped = store.isFinished();
    ShowWrappedResult< int >* wrapper = new ShowWrappedResult<int>( &wrapped );
    store.isFinished( wrapper );

    // Job< ParallelApplyData<int> > job( assign, 0, store );
    ParallelApply<int> job( assign, 0, store );
    job.run();

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

    return 0;
}

