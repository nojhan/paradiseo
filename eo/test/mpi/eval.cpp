//-----------------------------------------------------------------------------
// t-eoMpiParallel.cpp
//-----------------------------------------------------------------------------

#include <eo>
#include <eoPopEvalFunc.h>

#include <es/make_real.h>
// #include <apply.h>
#include "../real_value.h"

#include <mpi/eoMpi.h>

#include <mpi/eoTerminateJob.h>

#include <boost/mpi.hpp>

#include <vector>
using namespace std;

//-----------------------------------------------------------------------------

class eoRealSerializable : public eoReal< eoMinimizingFitness >, public eoserial::Persistent
{
    public:

        eoRealSerializable(unsigned size = 0, double value = 0.0):
            eoReal<eoMinimizingFitness>(size, value) {}

        eoserial::Object* pack() const
        {
            eoserial::Object* obj = new eoserial::Object;
            obj->add( "array",
                    eoserial::makeArray< vector<double>, eoserial::MakeAlgorithm >
                    ( *this )
                    );
            return obj;
        }

        void unpack( const eoserial::Object* obj )
        {
            eoserial::unpackArray< vector<double>, eoserial::Array::UnpackAlgorithm >
                ( *obj, "array", *this );
        }

        // Gives access to boost serialization
        friend class boost::serialization::access;

        /**
         * Serializes the decomposition in a boost archive (useful for boost::mpi)
         */
        template <class Archive>
            void save( Archive & ar, const unsigned int version ) const
            {
                std::stringstream ss;
                printOn( ss );
                std::string asStr = ss.str();
                ar & asStr;

                (void) version; // avoid compilation warning
            }

        /**
         * Deserializes the decomposition from a boost archive (useful for boost:mpi)
         */
        template <class Archive>
            void load( Archive & ar, const unsigned int version )
            {
                std::string asStr;
                ar & asStr;
                std::stringstream ss;
                ss << asStr;
                readFrom( ss );

                (void) version; // avoid compilation warning
            }

        // Indicates that boost save and load operations are not the same.
        BOOST_SERIALIZATION_SPLIT_MEMBER()

};

typedef eoRealSerializable EOT;

int main(int ac, char** av)
{
    eo::mpi::Node::init( ac, av );
    eo::log << eo::setlevel( eo::debug );

    eoParser parser(ac, av);

    unsigned int popSize = parser.getORcreateParam((unsigned int)100, "popSize", "Population Size", 'P', "Evolution Engine").value();
    unsigned int dimSize = parser.getORcreateParam((unsigned int)10, "dimSize", "Dimension Size", 'd', "Evolution Engine").value();

    uint32_t seedParam = parser.getORcreateParam((uint32_t)0, "seed", "Random number seed", 0).value();
    if (seedParam == 0) { seedParam = time(0); }

    make_parallel(parser);
    make_help(parser);

    rng.reseed( seedParam );

    eoUniformGenerator< double > gen(-5, 5);
    eoInitFixedLength< EOT > init( dimSize, gen );

    eoEvalFuncPtr< EOT, double, const std::vector< double >& > mainEval( real_value );
    eoEvalFuncCounter< EOT > eval( mainEval );

    eoPop< EOT > pop( popSize, init );

    eo::log << "Size of population : " << popSize << std::endl;

    eo::log << eo::setlevel( eo::debug );

    eo::mpi::DynamicAssignmentAlgorithm assign;
    eoParallelPopLoopEval< EOT > popEval( eval, assign, eo::mpi::DEFAULT_MASTER, 3 );
    popEval( pop, pop );

    eo::log << eo::quiet << "DONE!" << std::endl;

    return 0;
}

//-----------------------------------------------------------------------------
