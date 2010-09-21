#include <sstream>
#include <iomanip>
#include <fstream>

#include <eo>
#include <mo>

#include <utils/eoLogger.h>
#include <utils/eoParserLogger.h>

#include <do>

#include "Rosenbrock.h"
#include "Sphere.h"

typedef eoReal< eoMinimizingFitness > EOT;
typedef doNormalMulti< EOT > Distrib;
typedef EOT::AtomType AtomType;

int main(int ac, char** av)
{
    //-----------------------------------------------------
    // (0) parser + eo routines
    //-----------------------------------------------------

    eoParserLogger parser(ac, av);

    std::string	section("Algorithm parameters");

    unsigned int p_min = parser.createParam((unsigned int)10, "population-min", "Population min", 'p', section).value(); // p
    unsigned int p_max = parser.createParam((unsigned int)10000, "population-max", "Population max", 'P', section).value(); // P
    unsigned int p_step = parser.createParam((unsigned int)10, "population-step", "Population step", 't', section).value(); // t
    unsigned int s_size = parser.createParam((unsigned int)2, "dimension-size", "Dimension size", 'd', section).value(); // d

    AtomType mean_value = parser.createParam((AtomType)0, "mean", "Mean value", 'm', section).value(); // m
    AtomType covar1_value = parser.createParam((AtomType)1.0, "covar1", "Covar value 1", '1', section).value(); // 1
    AtomType covar2_value = parser.createParam((AtomType)0.5, "covar2", "Covar value 2", '2', section).value(); // 2
    AtomType covar3_value = parser.createParam((AtomType)1.0, "covar3", "Covar value 3", '3', section).value(); // 3

    if (parser.userNeedsHelp())
	{
	    parser.printHelp(std::cout);
	    exit(1);
	}

    make_verbose(parser);
    make_help(parser);

    //-----------------------------------------------------


    assert(s_size >= 2);

    eo::log << eo::debug << "p_size s_size mean(0) mean(1) new-mean(0) new-mean(1) distance" << std::endl;

    eo::log << eo::logging;

    for ( unsigned int p_size = p_min; p_size <= p_max; p_size *= p_step )
	{

	    assert(p_size >= p_min);

	    eoState state;


	    //-----------------------------------------------------
	    // (1) Population init and sampler
	    //-----------------------------------------------------

	    eoRndGenerator< double >* gen = new eoUniformGenerator< double >(-5, 5);
	    state.storeFunctor(gen);

	    eoInitFixedLength< EOT >* init = new eoInitFixedLength< EOT >( s_size, *gen );
	    state.storeFunctor(init);

	    // create an empty pop and let the state handle the memory
	    // fill population thanks to eoInit instance
	    eoPop< EOT >& pop = state.takeOwnership( eoPop< EOT >( p_size, *init ) );

	    //-----------------------------------------------------


	    //-----------------------------------------------------------------------------
	    // (2) distribution initial parameters
	    //-----------------------------------------------------------------------------

	    ublas::vector< AtomType > mean( s_size, mean_value );
	    ublas::symmetric_matrix< AtomType, ublas::lower > varcovar( s_size, s_size );

	    varcovar( 0, 0 ) = covar1_value;
	    varcovar( 0, 1 ) = covar2_value;
	    varcovar( 1, 1 ) = covar3_value;

	    Distrib distrib( mean, varcovar );

	    //-----------------------------------------------------------------------------


	    //-----------------------------------------------------------------------------
	    // Prepare bounder class to set bounds of sampling.
	    // This is used by doSampler.
	    //-----------------------------------------------------------------------------

	    doBounder< EOT >* bounder = new doBounderRng< EOT >(EOT(pop[0].size(), -5),
								EOT(pop[0].size(), 5),
								*gen);
	    state.storeFunctor(bounder);

	    //-----------------------------------------------------------------------------


	    //-----------------------------------------------------------------------------
	    // Prepare sampler class with a specific distribution
	    //-----------------------------------------------------------------------------

	    doSampler< Distrib >* sampler = new doSamplerNormalMulti< EOT >( *bounder );
	    state.storeFunctor(sampler);

	    //-----------------------------------------------------------------------------


	    //-----------------------------------------------------------------------------
	    // (4) sampling phase
	    //-----------------------------------------------------------------------------

	    pop.clear();

	    for (unsigned int i = 0; i < p_size; ++i)
		{
		    EOT candidate_solution = (*sampler)( distrib );
		    pop.push_back( candidate_solution );
		}

	    //-----------------------------------------------------------------------------


	    //-----------------------------------------------------------------------------
	    // (6) estimation phase
	    //-----------------------------------------------------------------------------

	    doEstimator< Distrib >* estimator = new doEstimatorNormalMulti< EOT >();
	    state.storeFunctor(estimator);

	    distrib = (*estimator)( pop );

	    //-----------------------------------------------------------------------------


	    //-----------------------------------------------------------------------------
	    // (8) euclidianne distance estimation
	    //-----------------------------------------------------------------------------

	    ublas::vector< AtomType > new_mean = distrib.mean();
	    ublas::symmetric_matrix< AtomType, ublas::lower > new_varcovar = distrib.varcovar();

	    AtomType distance = 0;

	    for ( unsigned int d = 0; d < s_size; ++d )
		{
		    distance += pow( mean[ d ] - new_mean[ d ], 2 );
		}

	    distance = sqrt( distance );

	    eo::log << p_size << " " << s_size << " "
		    << mean(0) << " " << mean(1) << " "
		    << new_mean(0) << " " << new_mean(1) << " "
		    << distance << std::endl
		;

	    //-----------------------------------------------------------------------------

	}

    return 0;
}
