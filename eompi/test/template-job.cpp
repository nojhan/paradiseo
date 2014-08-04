# include <eoMpi.h>

using namespace eo::mpi;

/*
 * This file is a template for a new eo::mpi::Job. You have everything that should be necessary to implement a new
 * parallelized algorithm.
 *
 * Replace __TEMPLATE__ by the name of your algorithm (for instance: MultiStart, ParallelApply, etc.).
 */

template< class EOT >
struct __TEMPLATE__Data
{

};

template< class EOT >
class SendTask__TEMPLATE__ : public SendTaskFunction< __TEMPLATE__Data< EOT > >
{
    public:

        using SendTaskFunction< __TEMPLATE__Data< EOT > >::_data;

        void operator()( int wrkRank )
        {
            // TODO implement me
        }
};

template< class EOT >
class HandleResponse__TEMPLATE__ : public HandleResponseFunction< __TEMPLATE__Data< EOT > >
{
    public:

        using HandleResponseFunction< __TEMPLATE__Data< EOT > >::_data;

        void operator()( int wrkRank )
        {
            // TODO implement me
        }
};

template< class EOT >
class ProcessTask__TEMPLATE__ : public ProcessTaskFunction< __TEMPLATE__Data< EOT > >
{
    public:
        using ProcessTaskFunction< __TEMPLATE__Data<EOT> >::_data;

        void operator()()
        {
            // TODO implement me
        }
};

template< class EOT >
class IsFinished__TEMPLATE__ : public IsFinishedFunction< __TEMPLATE__Data< EOT > >
{
    public:

        using IsFinishedFunction< __TEMPLATE__Data< EOT > >::_data;

        bool operator()()
        {
            // TODO implement me
        }
};

template< class EOT >
class __TEMPLATE__Store : public JobStore< __TEMPLATE__Data< EOT > >
{
    public:

        __TEMPLATE__Data<EOT>* data()
        {
            // TODO implement me
            return 0;
        }
};

template< class EOT >
class __TEMPLATE__ : public MultiJob< __TEMPLATE__Data< EOT > >
{
    public:

        __TEMPLATE__( AssignmentAlgorithm & algo,
                      int masterRank,
                      __TEMPLATE__Store< EOT > & store ) :
            MultiJob< __TEMPLATE__Data< EOT > >( algo, masterRank, store )
    {
        // TODO implement me
    }
};

/*
int main(int argc, char **argv)
{
    Node::init( argc, argv );

    DynamicAssignmentAlgorithm assignmentAlgo;
    __TEMPLATE__Store<int> store;
    __TEMPLATE__<int> job( assignmentAlgo, DEFAULT_MASTER, store );
}
*/
