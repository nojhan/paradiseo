/*
   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

 * Authors:
 *      Benjamin Bouvier <benjamin.bouvier@gmail.com>
 */

/**
 * @file t-mpi-distrib-exp.cpp
 * @brief File for parallel experimentations.
 *
 * When using parallel evaluation, the individuals to evaluate are sent by packets (group),
 * so as to avoid that communication time be more important than worker's execution time.
 * However, the ideal size of packet depends on the problem and the time needed to carry out
 * the atomic operation on each individual. This experiment tries to find a relation between
 * the total number of elements to process (size), the execution time and the size of packet.
 * This could lead to an heuristic allowing to optimize the size of packet according to the
 * processing times.
 */
# include <unistd.h> // usleep

# include <iostream>
# include <iomanip>
# include <string>
# include <sstream>
# include <vector>

# include <eo>

# include <mpi/eoMpi.h>
# include "t-mpi-common.h"

using namespace eo::mpi;

// Serializable int
typedef SerializableBase<int> type;

/*
 * The task is the following: the worker receives a number of milliseconds to wait, which
 * simulates the process of one individual. This way, the sequences of processing times are
 * generated only by the master and are more easily reproductible.
 */
struct Wait : public eoUF< type &, void >
{
    Wait( bool print ) : _print( print )
    {
        // empty
    }

    void operator()( type & milliseconds )
    {
        if( _print )
            std::cout << "Sleeping for " << milliseconds << "ms..." << std::endl;
        // usleep takes an input in microseconds
        usleep( milliseconds * 1000 );
    }

    private:
    bool _print;
};

/**
 * @brief Represents a distribution of processing times.
 */
class Distribution : public std::vector< type >, public eoserial::Persistent
{
    public:

    /**
     * @brief Really fills the vector with the distribution values.
     */
    void fill( unsigned size )
    {
        for( unsigned i = 0; i < size; ++i )
        {
            int next = next_element();
            if( next < 0 ) next = 0;
            push_back( next );
        }
    }

    /**
     * @brief Returns the next element of the distribution to put in the
     * vector.
     *
     * @returns Number of milliseconds to wait. Can be negative ; in this case,
     * the number will be truncated to 0ms.
     */
    virtual int next_element() = 0;

    /**
     * @brief Creates params and retrieves values from parser
     *
     * Parser's params should take milliseconds as inputs.
     */
    virtual void make_parser( eoParser & parser ) = 0;

    /**
     * @brief Returns true if this distribution has been activated by the
     * command line.
     *
     * Used by the main program so as to check if at least one distribution has been
     * activated.
     */
    bool isActive() { return _active; }

    /**
     * @brief Prints the name and the parameters of the distribution
     */
    virtual std::string toString() const = 0;

    protected:

    bool _active;
};

/**
 * @brief Uniform distribution.
 *
 * This is an uniform distribution, defined by a minimum value and a maximum value.
 * In the uniform distribution, every number from min to max has the same probability
 * to appear.
 *
 * The 3 parameters activable from a parser are the following:
 * - uniform=1 : if we want to use the uniform distribution
 * - uniform-min=x : use x milliseconds as the minimum value of waiting time.
 * - uniform-max=y : use y milliseconds as the maximum value of waiting time.
 * Ensure that x < y, or the results are unpredictable.
 */
class UniformDistribution : public Distribution
{
    public:

    UniformDistribution()
    {
        // empty
    }

    void make_parser( eoParser & parser )
    {
        _active = parser.createParam( false, "uniform", "Uniform distribution", '\0', "Uniform").value();
        _min = parser.createParam( 0.0, "uniform-min", "Minimum for uniform distribution, in ms.", '\0', "Uniform").value();
        _max = parser.createParam( 1.0, "uniform-max", "Maximum for uniform distribution, in ms.", '\0', "Uniform").value();
    }

    int next_element()
    {
        return std::floor( eo::rng.uniform( _min, _max ) );
    }

    eoserial::Object* pack( void ) const
    {
        eoserial::Object* obj = new eoserial::Object;
        obj->add( "name", eoserial::make( "uniform" ) );
        obj->add( "min", eoserial::make( _min ) );
        obj->add( "max", eoserial::make( _max ) );
        return obj;
    }

    void unpack( const eoserial::Object* obj )
    {
        eoserial::unpack( *obj, "min", _min );
        eoserial::unpack( *obj, "max", _max );
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss << "uniform" << '\n'
            << "min: " << _min << '\n'
            << "max: " << _max << '\n';
        return ss.str();
    }

    protected:

    double _min;
    double _max;

} uniformDistribution;

/**
 * @brief Normal (gaussian) distribution of times.
 *
 * A normal distribution is defined by a mean and a standard deviation.
 * The 3 parameters activable from the parser are the following:
 * - normal=1: activates the gaussian distribution.
 * - normal-mean=50: use 50ms as the mean of the distribution.
 * - normal-stddev=10: use 10ms as the standard deviation of the distribution.
 */
class NormalDistribution : public Distribution
{
    public:

    NormalDistribution()
    {
        // empty
    }

    void make_parser( eoParser & parser )
    {
        _active = parser.createParam( false, "normal", "Normal distribution", '\0', "Normal").value();
        _mean = parser.createParam( 0.0, "normal-mean", "Mean for the normal distribution (0 by default), in ms.", '\0', "Normal").value();
        _stddev = parser.createParam( 1.0, "normal-stddev", "Standard deviation for the normal distribution (1ms by default), 0 isn't acceptable.", '\0', "Normal").value();
    }

    int next_element()
    {
        return std::floor( eo::rng.normal( _mean, _stddev ) );
    }

    eoserial::Object* pack( void ) const
    {
        eoserial::Object* obj = new eoserial::Object;
        obj->add( "name", eoserial::make( "normal" ) );
        obj->add( "mean", eoserial::make( _mean ) );
        obj->add( "stddev", eoserial::make( _stddev ) );
        return obj;
    }

    void unpack( const eoserial::Object* obj )
    {
        eoserial::unpack( *obj, "mean", _mean );
        eoserial::unpack( *obj, "stddev", _stddev );
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss << "normal" << '\n'
            << "mean: " << _mean << '\n'
            << "stddev: " << _stddev << '\n';
        return ss.str();
    }

    protected:

    double _mean;
    double _stddev;
} normalDistribution;

/**
 * @brief Exponential distribution.
 *
 * This distribution belongs to the category of the decreasing power laws and are affected by long trails
 * phenomenons.
 * An exponential distribution is only defined by its mean.
 *
 * The 2 parameters activable from the parser are the following:
 * - exponential=1: to activate the exponential distribution.
 * - exponential-mean=50: indicates that the mean must be 50ms.
 */
class ExponentialDistribution : public Distribution
{
    public:

    ExponentialDistribution()
    {
        // empty
    }

    void make_parser( eoParser & parser )
    {
        _active = parser.createParam( false, "exponential", "Exponential distribution", '\0', "Exponential").value();
        _mean = parser.createParam( 0.0, "exponential-mean", "Mean for the exponential distribution (0 by default), in ms.", '\0', "Exponential").value();
    }

    int next_element()
    {
        return std::floor( eo::rng.negexp( _mean ) );
    }

    eoserial::Object* pack( void ) const
    {
        eoserial::Object* obj = new eoserial::Object;
        obj->add( "name", eoserial::make( "exponential" ) );
        obj->add( "mean", eoserial::make( _mean ) );
        return obj;
    }

    void unpack( const eoserial::Object* obj )
    {
        eoserial::unpack( *obj, "mean", _mean );
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss << "exponential" << '\n'
            << "mean: " << _mean << '\n';
        return ss.str();
    }

    protected:

    double _mean;

} exponentialDistribution;

/**
 * @brief Serializable experiment.
 *
 * Allows an experiment to be saved and loaded via a file, using eoserial.
 *
 * Construct the experiment with the good parameters from the command line or load experiments from a file. Then call run() to launch the parallel job.
 *
 * If a filename is given to the constructor (or during the loading), the results of the experiments (time series) will
 * be redirected to the file with the given file name. Otherwise (filename == ""), the output will just be shown on the
 * standard output.
 */
class Experiment : public eoserial::Persistent
{
    public:

    Experiment() : _distribution(0), _worker_print_waiting_time( false ), _fileName("")
    {
        // empty
    }

    Experiment( Distribution* distrib, unsigned size, unsigned packet_size, bool print_waiting_time, unsigned seed, const std::string& fileName = "" ) :
        _distribution( distrib ),
        _size( size ),
        _packet_size( packet_size ),
        _worker_print_waiting_time( print_waiting_time ),
        _seed( seed ),
        _fileName( fileName )
    {
        // empty
    }

    eoserial::Object* pack( void ) const
    {
        eoserial::Object* obj = new eoserial::Object;
        obj->add( "size", eoserial::make( _size ) );
        obj->add( "packet_size", eoserial::make( _packet_size ) );
        obj->add( "worker_print_waiting_time", eoserial::make( _worker_print_waiting_time ) );
        obj->add( "seed", eoserial::make( _seed ) );
        if( _distribution )
        {
            obj->add( "distribution", _distribution );
        }
        obj->add( "filename", eoserial::make( _fileName ) );
        return obj;
    }

    void unpack( const eoserial::Object* obj )
    {
        eoserial::unpack( *obj, "size", _size );
        eoserial::unpack( *obj, "packet_size", _packet_size );
        eoserial::unpack( *obj, "worker_print_waiting_time", _worker_print_waiting_time );
        eoserial::unpack( *obj, "seed", _seed );
        eoserial::unpack( *obj, "filename", _fileName );

        eoserial::Object* distribObject = static_cast<eoserial::Object*>( obj->find("distribution")->second );
        std::string distribName = *static_cast<eoserial::String*>( distribObject->find("name")->second );

        // TODO find a better design...
        if( distribName == "normal" ) {
            _distribution = & normalDistribution;
        } else if( distribName == "uniform" ) {
            _distribution = & uniformDistribution;
        } else if( distribName == "exponential" ) {
            _distribution = & exponentialDistribution;
        } else {
            throw eoException("When unpacking experience, no distribution found.");
        }

        eoserial::unpackObject( *obj, "distribution", *_distribution );
    }

    void run()
    {
        communicator& comm = eo::mpi::Node::comm();
        // reinits every objects
        eo::rng.reseed( _seed );
        eo::rng.clearCache(); // trick for repeatable sequences of normal numbers, cf eo::rng
        _distribution->clear();
        _distribution->fill( _size );

        eo::mpi::timerStat.start("run");
        Wait wait( _worker_print_waiting_time );
        ParallelApplyStore< type > store( wait, DEFAULT_MASTER, _packet_size );
        store.data( *_distribution );
        DynamicAssignmentAlgorithm scheduling;
        ParallelApply< type > job( scheduling, DEFAULT_MASTER, store );

        job.run();
        eo::mpi::timerStat.stop("run");
        if( job.isMaster() )
        {
            EmptyJob( scheduling, DEFAULT_MASTER ); // to terminate parallel apply
            // Receive statistics
            typedef std::map< std::string, eoTimerStat::Stat > typeStats;

            std::ostream* pout;
            std::ofstream file;
            bool fileSaveActivated = false;
            if( _fileName == "" ) {
                pout = & std::cout;
            } else {
                pout = & file;
                file.open( _fileName.c_str() );
                fileSaveActivated = true;
            }
            std::ostream& out = *pout;

            // Reminder of the parameters
            out << "size: " << _size << '\n'
                << "packet_size: " << _packet_size << '\n'
                << "distribution: " << _distribution->toString()
                << "seed: " << _seed << '\n' << std::endl;

            // Results
            out << std::fixed << std::setprecision( 5 );
            for( int i = 1, s = comm.size(); i < s; ++i )
            {
                eoTimerStat timerStat;
                comm.recv( i, eo::mpi::Channel::Commands, timerStat );
                typeStats stats = timerStat.stats();
                for( typeStats::iterator it = stats.begin(),
                        end = stats.end();
                        it != end;
                        ++it )
                {
                    out << i << " " << it->first << std::endl;
                    for( int j = 0, t = it->second.wtime.size(); j < t; ++j )
                    {
                        out << it->second.wtime[j] << " ";
                    }
                    out << std::endl;
                }
                out << std::endl;
            }

            if( fileSaveActivated ) {
                file.close();
            }
        } else
        {
            // Send statistics
            comm.send( DEFAULT_MASTER, eo::mpi::Channel::Commands, eo::mpi::timerStat );
        }
        timerStat.clear();
    }

    private:

    Distribution* _distribution;
    unsigned _size;
    unsigned _packet_size;
    bool _worker_print_waiting_time;
    unsigned _seed;
    std::string _fileName;
};

int main( int argc, char** argv )
{
    Node::init( argc, argv );
    eoParser parser( argc, argv );

    // forces the statistics to be retrieved
    eo::mpi::timerStat.forceDoMeasure();

    // General parameters for the experimentation
    unsigned size = parser.createParam( 10U, "size", "Number of elements to distribute.", 's', "Distribution").value();
    unsigned packet_size = parser.createParam( 1U, "packet-size", "Number of elements to distribute at each time for a single worker.", 'p', "Parallelization").value();
    bool worker_print_waiting_time = parser.createParam( false, "print-waiting-time", "Do the workers need to print the time they wait?", '\0', "Parallelization").value();
    unsigned seed = parser.createParam( 0U, "seed", "Seed of random generator", '\0', "General").value();
    std::string fileName = parser.createParam( std::string(""), "filename", "File name to which redirect the results (for a single experiment)", '\0', "General").value();

    bool useExperimentFile = parser.createParam( false, "use-experiment-file", "Put to true if you want to launch experiments from a file formatted in JSON (see experiment-file).", '\0', "General").value();
    std::string experimentFile = parser.createParam( std::string("experiments.json"), "experiment-file", "File name of experiments to provide, in format JSON.", '\0', "General").value();

    if( !useExperimentFile )
    {
        std::vector<Distribution*> distribs;
        distribs.push_back( &uniformDistribution );
        distribs.push_back( &normalDistribution );
        distribs.push_back( &exponentialDistribution );

        // for each available distribution, check if activated.
        // If no distribution is activated, show an error message
        // If two distributions or more are activated, show an error message
        // Otherwise, use the activated distribution as distrib
        bool isChosenDistrib = false;
        Distribution* pdistrib = 0;
        for( int i = 0, s = distribs.size(); i < s; ++i )
        {
            distribs[i]->make_parser( parser );
            if( distribs[i]->isActive() )
            {
                if( isChosenDistrib )
                {
                    throw eoException("Only one distribution can be chosen during a launch!");
                } else
                {
                    isChosenDistrib = true;
                    pdistrib = distribs[i];
                }
            }
        }

        make_parallel( parser );
        make_help( parser );

        if( !isChosenDistrib )
        {
            throw eoException("No distribution chosen. One distribution should be chosen.");
        }

        Experiment e( pdistrib, size, packet_size, worker_print_waiting_time, seed, fileName );
        e.run();
    }
    else // use experiments file
    {
        // read content of file
        std::ifstream file( experimentFile.c_str() );
        std::string fileContent;
        while( file )
        {
            char temp[4096];
            file.getline( temp, 4096, '\n' );
            fileContent += temp;
            fileContent += '\n';
        }
        file.close();

        // transform content into array of experiments
        eoserial::Object* wrapper = eoserial::Parser::parse( fileContent );
        eoserial::Array& experiments = *static_cast< eoserial::Array* >( wrapper->find("experiments")->second );

        for( unsigned i = 0, s = experiments.size(); i < s; ++i )
        {
            std::cout << "Launching experiment " << (i+1) << "..." << std::endl;
            eoserial::Object* expObj = static_cast< eoserial::Object* >( experiments[i] );
            Experiment exp;
            exp.unpack( expObj );
            exp.run();
        }
        delete wrapper;
    }

    return 0;
}
