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

    unsigned int p_size = parser.createParam((unsigned int)100, "popSize", "Population Size", 'P', section).value(); // P

    unsigned int s_size = parser.createParam((unsigned int)2, "dimension-size", "Dimension size", 'd', section).value(); // d

    AtomType mean_value = parser.createParam((AtomType)0, "mean", "Mean value", 'm', section).value(); // m

    AtomType covar1_value = parser.createParam((AtomType)1, "covar1", "Covar value 1", '1', section).value();
    AtomType covar2_value = parser.createParam((AtomType)0.5, "covar2", "Covar value 2", '2', section).value();
    AtomType covar3_value = parser.createParam((AtomType)1, "covar3", "Covar value 3", '3', section).value();

    if (parser.userNeedsHelp())
	{
	    parser.printHelp(std::cout);
	    exit(1);
	}

    make_verbose(parser);
    make_help(parser);


    assert(p_size > 0);
    assert(s_size > 0);


    eoState state;

    //-----------------------------------------------------


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
    // (3) distribution output
    //-----------------------------------------------------------------------------

    doDummyContinue< Distrib >* dummy_continue = new doDummyContinue< Distrib >();
    state.storeFunctor(dummy_continue);

    doCheckPoint< Distrib >* distribution_continue = new doCheckPoint< Distrib >( *dummy_continue );
    state.storeFunctor(distribution_continue);

    doDistribStat< Distrib >* distrib_stat = new doStatNormalMulti< EOT >();
    state.storeFunctor(distrib_stat);

    distribution_continue->add( *distrib_stat );

    eoMonitor* stdout_monitor = new eoStdoutMonitor();
    state.storeFunctor(stdout_monitor);
    stdout_monitor->add(*distrib_stat);
    distribution_continue->add( *stdout_monitor );

    (*distribution_continue)( distrib );

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

    // pop.sort();

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // (5) population output
    //-----------------------------------------------------------------------------

    eoContinue< EOT >* cont = new eoGenContinue< EOT >( 2 ); // never reached fitness
    state.storeFunctor(cont);

    eoCheckPoint< EOT >* pop_continue = new eoCheckPoint< EOT >( *cont );
    state.storeFunctor(pop_continue);

    doPopStat< EOT >* popStat = new doPopStat<EOT>;
    state.storeFunctor(popStat);
    pop_continue->add(*popStat);

    doFileSnapshot* fileSnapshot = new doFileSnapshot("TestResPop");
    state.storeFunctor(fileSnapshot);
    fileSnapshot->add(*popStat);
    pop_continue->add(*fileSnapshot);

    (*pop_continue)( pop );

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // (6) estimation phase
    //-----------------------------------------------------------------------------

    doEstimator< Distrib >* estimator = new doEstimatorNormalMulti< EOT >();
    state.storeFunctor(estimator);

    distrib = (*estimator)( pop );

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // (7) distribution output
    //-----------------------------------------------------------------------------

    (*distribution_continue)( distrib );

    //-----------------------------------------------------------------------------


    return 0;
}
