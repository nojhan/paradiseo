#include <eo>
#include <mo>

#include <utils/eoLogger.h>
#include <utils/eoParserLogger.h>

#include <do/make_pop.h>
#include <do/make_run.h>
#include <do/make_continue.h>
#include <do/make_checkpoint.h>

#include <do>

#include "Rosenbrock.h"
#include "Sphere.h"

typedef eoReal<eoMinimizingFitness> EOT;

//typedef doUniform< EOT > Distrib;
//typedef doNormalMono< EOT > Distrib;
typedef doNormalMulti< EOT > Distrib;

int main(int ac, char** av)
{
    eoParserLogger	parser(ac, av);

    // Letters used by the following declarations :
    // a d i p t

    std::string	section("Algorithm parameters");

    // FIXME: a verifier la valeur par defaut
    double initial_temperature = parser.createParam((double)10e5, "temperature", "Initial temperature", 'i', section).value(); // i

    eoState state;

    //-----------------------------------------------------------------------------
    // Instantiate all need parameters for EDASA algorithm
    //-----------------------------------------------------------------------------

    eoSelect< EOT >* selector = new eoDetSelect< EOT >(0.5);
    state.storeFunctor(selector);

    doEstimator< Distrib >* estimator =
	//new doEstimatorUniform< EOT >();
	//new doEstimatorNormalMono< EOT >();
	new doEstimatorNormalMulti< EOT >();
    state.storeFunctor(estimator);

    eoSelectOne< EOT >* selectone = new eoDetTournamentSelect< EOT >( 2 );
    state.storeFunctor(selectone);

    doModifierMass< Distrib >* modifier =
	//new doUniformCenter< EOT >();
	//new doNormalMonoCenter< EOT >();
	new doNormalMultiCenter< EOT >();
    state.storeFunctor(modifier);

    eoEvalFunc< EOT >* plainEval =
	new Rosenbrock< EOT >();
    //new Sphere< EOT >();
    state.storeFunctor(plainEval);

    unsigned long max_eval = parser.getORcreateParam((unsigned long)0, "maxEval", "Maximum number of evaluations (0 = none)", 'E', "Stopping criterion").value(); // E
    eoEvalFuncCounter< EOT > eval(*plainEval, max_eval);

    eoRndGenerator< double >* gen = new eoUniformGenerator< double >(-5, 5);
    //eoRndGenerator< double >* gen = new eoNormalGenerator< double >(0, 1);
    state.storeFunctor(gen);


    unsigned int dimension_size = parser.createParam((unsigned int)10, "dimension-size", "Dimension size", 'd', section).value(); // d

    eoInitFixedLength< EOT >* init = new eoInitFixedLength< EOT >( dimension_size, *gen );
    state.storeFunctor(init);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // (1) Population init and sampler
    //-----------------------------------------------------------------------------

    // Generation of population from do_make_pop (creates parameter, manages persistance and so on...)
    // ... and creates the parameter letters: L P r S

    // this first sampler creates a uniform distribution independently of our distribution (it doesnot use doUniform).

    eoPop< EOT >& pop = do_make_pop(parser, state, *init);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // (2) First evaluation before starting the research algorithm
    //-----------------------------------------------------------------------------

    apply(eval, pop);

    //-----------------------------------------------------------------------------


    //doBounder< EOT >* bounder = new doBounderNo< EOT >();
    doBounder< EOT >* bounder = new doBounderRng< EOT >(EOT(pop[0].size(), -5),
							EOT(pop[0].size(), 5),
							*gen);
    state.storeFunctor(bounder);

    doSampler< Distrib >* sampler =
	//new doSamplerUniform< EOT >();
	//new doSamplerNormalMono< EOT >( *bounder );
	new doSamplerNormalMulti< EOT >( *bounder );
    state.storeFunctor(sampler);


    // FIXME: should I set the default value of rho to pop size ?!?

    unsigned int rho = parser.createParam((unsigned int)0, "rho", "Rho: metropolis sample size", 'p', section).value(); // p

    moGenSolContinue< EOT >* sa_continue = new moGenSolContinue< EOT >(rho);
    state.storeFunctor(sa_continue);

    double threshold = parser.createParam((double)0.1, "threshold", "Threshold: temperature threshold stopping criteria", 't', section).value(); // t
    double alpha = parser.createParam((double)0.1, "alpha", "Alpha: temperature dicrease rate", 'a', section).value(); // a

    moCoolingSchedule* cooling_schedule = new moGeometricCoolingSchedule(threshold, alpha);
    state.storeFunctor(cooling_schedule);

    // stopping criteria
    // ... and creates the parameter letters: C E g G s T

    eoContinue< EOT >& eo_continue = do_make_continue(parser, state, eval);

    // output

    eoCheckPoint< EOT >& monitoring_continue = do_make_checkpoint(parser, state, eval, eo_continue);

    // eoPopStat< EOT >* popStat = new eoPopStat<EOT>;
    // state.storeFunctor(popStat);

    // checkpoint.add(*popStat);

    // eoGnuplot1DMonitor* gnuplot = new eoGnuplot1DMonitor("gnuplot.txt");
    // state.storeFunctor(gnuplot);

    // gnuplot->add(eval);
    // gnuplot->add(*popStat);

    //gnuplot->gnuplotCommand("set yrange [0:500]");

    // checkpoint.add(*gnuplot);

    // eoMonitor* fileSnapshot = new doFileSnapshot< std::vector< std::string > >("ResPop");
    // state.storeFunctor(fileSnapshot);

    // fileSnapshot->add(*popStat);
    // checkpoint.add(*fileSnapshot);


    doDummyContinue< Distrib >* dummy_continue = new doDummyContinue< Distrib >();
    state.storeFunctor(dummy_continue);

    doCheckPoint< Distrib >* distribution_continue = new doCheckPoint< Distrib >( *dummy_continue );
    state.storeFunctor(distribution_continue);

    doDistribStat< Distrib >* distrib_stat = new doStatNormalMulti< EOT >();
    state.storeFunctor(distrib_stat);

    distribution_continue->add( *distrib_stat );

    // eoMonitor* stdout_monitor = new eoStdoutMonitor();
    // state.storeFunctor(stdout_monitor);
    // stdout_monitor->add(*distrib_stat);
    // distribution_continue->add( *stdout_monitor );

    eoFileMonitor* file_monitor = new eoFileMonitor("distribution_bounds.txt");
    state.storeFunctor(file_monitor);
    file_monitor->add(*distrib_stat);
    distribution_continue->add( *file_monitor );

    //-----------------------------------------------------------------------------
    // eoEPRemplacement causes the using of the current and previous
    // sample for sampling.
    //-----------------------------------------------------------------------------

    eoReplacement< EOT >* replacor = new eoEPReplacement< EOT >(pop.size());

    // Below, use eoGenerationalReplacement to sample only on the current sample

    //eoReplacement< EOT >* replacor = new eoGenerationalReplacement< EOT >(); // FIXME: to define the size

    state.storeFunctor(replacor);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // EDASA algorithm configuration
    //-----------------------------------------------------------------------------

    doAlgo< Distrib >* algo = new doEDASA< Distrib >
    	(*selector, *estimator, *selectone, *modifier, *sampler,
    	 monitoring_continue, *distribution_continue,
	 eval, *sa_continue, *cooling_schedule,
    	 initial_temperature, *replacor);

    //-----------------------------------------------------------------------------


    // state.storeFunctor(algo);

    if (parser.userNeedsHelp())
	{
	    parser.printHelp(std::cout);
	    exit(1);
	}

    // Help + Verbose routines

    make_verbose(parser);
    make_help(parser);


    //-----------------------------------------------------------------------------
    // Beginning of the algorithm call
    //-----------------------------------------------------------------------------

    try
	{
	    do_run(*algo, pop);
	}
    catch (eoReachedThresholdException& e)
	{
	    eo::log << eo::warnings << e.what() << std::endl;
	}
    catch (std::exception& e)
	{
	    eo::log << eo::errors << "exception: " << e.what() << std::endl;
	    exit(EXIT_FAILURE);
	}

    //-----------------------------------------------------------------------------

    return 0;
}
