# include <unistd.h> // usleep

# include <iostream>
# include <string>
# include <vector>

# include <eo>

# include <mpi/eoParallelApply.h>
# include "../test/mpi/t-mpi-common.h"

using namespace eo::mpi;

typedef SerializableBase<int> type;

struct Wait : public eoUF< type &, void >
{
    void operator()( type & milliseconds )
    {
        std::cout << "Sleeping for " << milliseconds << "ms..." << std::endl;
        // usleep takes an input in microseconds
        usleep( milliseconds * 1000 );
    }
} wait;

class Distribution : public std::vector< type >
{
    public:

    /**
     * @brief Really fills the vector with the distribution values.
     */
    void fill( unsigned size )
    {
        for( unsigned i = 0; i < size; ++i )
        {
            push_back( next_element() );
        }
    }

    /**
     * @brief Returns the next element of the distribution to put in the
     * vector.
     *
     * @returns Number of milliseconds to wait
     */
    virtual int next_element() = 0;

    // Idea for function name: enlarge_your_parser
    /**
     * @brief Creates params and retrieves values from parser
     */
    virtual void make_parser( eoParser & parser ) = 0;

    /**
     * @brief Returns true if this distribution has been activated by the
     * command line.
     *
     * Serves to main program to check if at least one distribution has been
     * activated.
     */
    bool isActive() { return _active; }

    protected:

    bool _active;
};

class UniformDistribution : public Distribution
{
    public:

    UniformDistribution() : _rng(0)
    {
        // empty
    }

    void make_parser( eoParser & parser )
    {
        _active = parser.createParam( false, "uniform", "Uniform distribution", '\0', "Uniform").value();
        _min = parser.createParam( 0.0, "uniform-min", "Minimum for uniform distribution", '\0', "Uniform").value();
        _max = parser.createParam( 1.0, "uniform-max", "Maximum for uniform distribution", '\0', "Uniform").value();
    }

    int next_element()
    {
        return std::floor( 1000. * _rng.uniform( _min, _max ) );
    }

    protected:

    eoRng _rng;

    double _min;
    double _max;

} uniformDistribution;

int main( int argc, char** argv )
{
    Node::init( argc, argv );
    eoParser parser( argc, argv );

    // TODO for each available distribution, check if activated.
    // If no distribution is activated, show an error message
    // If two distributions or more are activated, show an error message
    // Otherwise, use the activated distribution as distrib
    Distribution & distrib = uniformDistribution;
    // Make parser of distribution here
    distrib.make_parser( parser );

    unsigned size = parser.createParam( 10U, "size", "Number of elements to distribute.", 's', "Distribution").value();
    unsigned packet_size = parser.createParam( 1U, "packet_size", "Number of elements to distribute at each time for a single worker.", 'p', "Parallelization").value();

    make_parallel( parser );
    make_help( parser );

    ParallelApplyStore< type> store( wait, DEFAULT_MASTER, packet_size );

    // Fill distribution
    distrib.fill( size );
    store.data( distrib );

    DynamicAssignmentAlgorithm scheduling;
    ParallelApply< type > job( scheduling, DEFAULT_MASTER, store );

    job.run();

    if( job.isMaster() )
    {
        EmptyJob( scheduling, DEFAULT_MASTER ); // to terminate parallel apply
    }

    return 0;
}
