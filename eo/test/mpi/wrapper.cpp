# include <mpi/eoMpi.h>
# include <mpi/eoParallelApply.h>
# include <mpi/eoTerminateJob.h>

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

    ParallelApplyStore< int > store( plusOneInstance, eo::mpi::DEFAULT_MASTER, 1 );
    store.data( v );
    store.wrapIsFinished( new ShowWrappedResult<int> );

    ParallelApply<int> job( assign, eo::mpi::DEFAULT_MASTER, store );
    // Equivalent to:
    // Job< ParallelApplyData<int> > job( assign, 0, store );
    job.run();
    EmptyJob stop( assign, eo::mpi::DEFAULT_MASTER );

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

