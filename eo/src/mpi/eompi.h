# ifndef __EO_MPI_H__
# define __EO_MPI_H__

# include <vector>
# include <map>

# include <boost/mpi.hpp>
namespace mpi = boost::mpi;

# include <iostream>
using namespace std;
// TODO TODOB comment!

namespace EoMpi
{
    namespace Channel
    {
        const int Commands = 0;
    }

    namespace Message
    {
        const int Continue = 0;
        const int Finish = 1;
    }
}

class MpiNode;

class MpiNodeStore
{
    public:

    static void instance( MpiNode* _instance )
    {
        singleton = _instance;
    }

    static MpiNode* instance()
    {
        return singleton;
    }

    protected:

    static MpiNode* singleton;
};

class MpiNode
{
protected:
    mpi::environment& env;
    mpi::communicator& _comm;

    int rank;
    int size;

    int argc;
    char** argv;

public:
    MpiNode( mpi::environment& _env, mpi::communicator& __comm ) :
        env(_env),
        _comm(__comm),
        rank(__comm.rank()),
        size(__comm.size())
    {
        // empty
    }

    virtual ~MpiNode()
    {
        // empty
    }

    mpi::communicator& comm()
    {
        return _comm;
    }
};

struct AssignmentAlgorithm
{
    virtual int get( ) = 0;
    virtual void size( int s ) = 0;
    virtual int size( ) = 0;
    virtual void confirm( int wrkRank ) = 0;
};

struct DynamicAssignmentAlgorithm : public AssignmentAlgorithm
{
    public:
        virtual int get( )
        {
            int assignee = -1;
            if (! availableWrk.empty() )
            {
                assignee = availableWrk.back();
                availableWrk.pop_back();
            }
            return assignee;
        }
;
        void size( int s )
        {
            for( int i = 1; i < s ; ++i )
            {
                availableWrk.push_back( i );
            }
        }

        int size()
        {
            return availableWrk.size();
        }

        void confirm( int rank )
        {
            availableWrk.push_back( rank );
        }

    protected:
        std::vector< int > availableWrk;
};

template< typename EOT >
class MpiJob
{
    public:

    MpiJob( std::vector< EOT > & _data ) :
        data( _data ),
        comm( MpiNodeStore::instance()->comm() )
    {
        // empty
    }

    // master
    virtual void sendTask( int wrkRank, int index ) = 0;
    virtual void handleResponse( int wrkRank, int index ) = 0;
    // worker
    virtual void processTask( ) = 0;

    void master( AssignmentAlgorithm & assignmentAlgorithm )
    {
        for( int i = 0, size = data.size();
                i < size;
                ++i)
        {
            cout << "Beginning loop for i = " << i << endl;
            int assignee = assignmentAlgorithm.get( );
            cout << "Assignee : " << assignee << endl;
            while( assignee <= 0 )
            {
                cout << "Waitin' for node..." << endl;
                mpi::status status = comm.probe( mpi::any_source, mpi::any_tag );
                int wrkRank = status.source();
                cout << "Node " << wrkRank << " just terminated." << endl;
                handleResponse( wrkRank, assignedTasks[ wrkRank ] );
                assignmentAlgorithm.confirm( wrkRank );
                assignee = assignmentAlgorithm.get( );
            }
            cout << "Assignee found : " << assignee << endl;
            assignedTasks[ assignee ] = i;
            comm.send( assignee, EoMpi::Channel::Commands, EoMpi::Message::Continue );
            sendTask( assignee, i );
        }

        // frees all the idle workers
        int idle;
        vector<int> idles;
        while ( ( idle = assignmentAlgorithm.get( ) ) > 0 )
        {
            comm.send( idle, EoMpi::Channel::Commands, EoMpi::Message::Finish );
            idles.push_back( idle );
        }
        for (int i = 0; i < idles.size(); ++i)
        {
            assignmentAlgorithm.confirm( idles[i] );
        }

        // wait for all responses
        int wrkNb = comm.size() - 1;
        while( assignmentAlgorithm.size() != wrkNb )
        {
            mpi::status status = comm.probe( mpi::any_source, mpi::any_tag );
            int wrkRank = status.source();
            handleResponse( wrkRank, assignedTasks[ wrkRank ] );
            comm.send( wrkRank, EoMpi::Channel::Commands, EoMpi::Message::Finish );
            assignmentAlgorithm.confirm( wrkRank );
        }
    }

    void worker( )
    {
        int order;
        while( true )
        {
            comm.recv( 0, EoMpi::Channel::Commands, order );
            if ( order == EoMpi::Message::Finish )
            {
                return;
            } else
            {
                processTask( );
            }
        }
    }

protected:

    std::vector<EOT> & data;
    std::map< int /* worker rank */, int /* index in vector */> assignedTasks;

    mpi::communicator& comm;
};

class MasterNode : public MpiNode
{
public:
    MasterNode( int _argc, char** _argv,
            mpi::environment& _env,
            mpi::communicator& _comm
            ) :
        MpiNode(_env, _comm )
    {
        // empty
    }

    void setAssignmentAlgorithm( AssignmentAlgorithm* assignmentAlgo )
    {
        _assignmentAlgo = assignmentAlgo;
        _assignmentAlgo->size( _comm.size() );
    }

    template< typename EOT >
    void run( MpiJob< EOT > & job )
    {
        job.master( *_assignmentAlgo );
    }

protected:
    AssignmentAlgorithm* _assignmentAlgo;
};

class WorkerNode : public MpiNode
{
    public:

        WorkerNode(
                int _argc, char** _argv,
                mpi::environment& _env,
                mpi::communicator& _comm ) :
            MpiNode( _env, _comm )
        {
            // empty
        }

        template< typename EOT >
        void run( MpiJob<EOT> & job )
        {
            job.worker( );
        }
};

class MpiSingletonFactory
{
    public:

    static void init( int argc, char** argv )
    {
        MpiNode* singleton;
        //mpi::environment* env = new mpi::environment ( argc, argv );
        //mpi::communicator* world = new mpi::communicator; // TODO clean
        static mpi::environment env( argc, argv );
        static mpi::communicator world;
        if ( world.rank() == 0 )
        {
            singleton = new MasterNode( argc, argv, env, world );
        } else
        {
            singleton = new WorkerNode( argc, argv, env, world );
        }
        MpiNodeStore::instance( singleton );
    }
};
# endif // __EO_MPI_H__
