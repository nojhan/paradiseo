//-----------------------------------------------------------------------------
// t-openmp.cpp
//-----------------------------------------------------------------------------

#include <utils/eoLogger.h>
#include <utils/eoParserLogger.h>

#include <eo>
#include <es/make_real.h>

#include <apply.h>
#include <omp_apply.h>

#include <omp.h>

#include "real_value.h"

//-----------------------------------------------------------------------------

typedef eoReal< eoMinimizingFitness > EOT;

//-----------------------------------------------------------------------------

int main(int ac, char** av)
{
    eoParserLogger parser(ac, av);
    eoState state;

    eoRealInitBounded<EOT>& init = make_genotype( parser, state, EOT() );
    eoPop< EOT >& pop = make_pop( parser, state, init );
    eoPop< EOT >& pop2 = make_pop( parser, state, init );
    eoEvalFuncPtr< EOT, double, const std::vector< double >& > mainEval( real_value );
    eoEvalFuncCounter<EOT> eval( mainEval );

    if (parser.userNeedsHelp())
	{
	    parser.printHelp(std::cout);
	    exit(1);
	}

    make_help(parser);
    make_verbose(parser);

    double ts1 = omp_get_wtime();
    apply< EOT >( eval, pop );
    //sleep(1);
    double ts2 = omp_get_wtime();

    double tp1 = omp_get_wtime();
    omp_apply< EOT >( eval, pop2 );
    //sleep(1);
    double tp2 = omp_get_wtime();

    eo::log << "Ts = " << ts2 - ts1 << std::endl;
    eo::log << "Tp = " << tp2 - tp1 << std::endl;

    return 0;
}

//-----------------------------------------------------------------------------
