//-----------------------------------------------------------------------------
// t-eoParallel.cpp
//-----------------------------------------------------------------------------

#include <omp.h>

#include <eo>
#include <es/make_real.h>
//#include <apply.h>
#include "real_value.h"

//-----------------------------------------------------------------------------

typedef eoReal< eoMinimizingFitness > EOT;

int main(int ac, char** av)
{
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

    //apply< EOT >( eval, pop );
    eoPopLoopEval< EOT > popEval( eval );
    popEval( pop, pop );

    eo::log << eo::quiet << "DONE!" << std::endl;

#pragma omp parallel
    {
	if ( 0 == omp_get_thread_num() )
	    {
		eo::log << "num of threads: " << omp_get_num_threads() << std::endl;
	    }
    }

    return 0;
}

//-----------------------------------------------------------------------------
