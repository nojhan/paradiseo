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

typedef eoReal<eoMinimizingFitness>	EOT;

int	main(int ac, char** av)
{
    eoParserLogger	parser(ac, av);

    // Letters used by the following declarations :
    // a d i p t

    std::string	section("Algorithm parameters");

    // FIXME: a verifier la valeur par defaut
    double initial_temperature = parser.createParam((double)10e5, "temperature", "Initial temperature", 'i', section).value(); // i

    eoState state;

    //-----------------------------------------------------------------------------
    // Instantiate all need parameters for CMASA algorithm
    //-----------------------------------------------------------------------------

    eoSelect< EOT >* selector = new eoDetSelect< EOT >(0.1);
    state.storeFunctor(selector);

    //doEstimator< doUniform< EOT > >* estimator = new doEstimatorUniform< EOT >();
    doEstimator< doNormal< EOT > >* estimator = new doEstimatorNormal< EOT >();
    state.storeFunctor(estimator);

    eoSelectOne< EOT >* selectone = new eoDetTournamentSelect< EOT >();
    state.storeFunctor(selectone);

    //doModifierMass< doUniform< EOT > >* modifier = new doUniformCenter< EOT >();
    doModifierMass< doNormal< EOT > >* modifier = new doNormalCenter< EOT >();
    state.storeFunctor(modifier);

    //eoEvalFunc< EOT >* plainEval = new BopoRosenbrock< EOT, double, const EOT& >();
    eoEvalFunc< EOT >* plainEval = new Sphere< EOT >();
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

    //doSampler< doUniform< EOT > >* sampler = new doSamplerUniform< EOT >();
    doSampler< doNormal< EOT > >* sampler = new doSamplerNormal< EOT >( *bounder );
    state.storeFunctor(sampler);


    unsigned int rho = parser.createParam((unsigned int)0, "rho", "Rho: metropolis sample size", 'p', section).value(); // p

    moGenSolContinue< EOT >* continuator = new moGenSolContinue< EOT >(rho);
    state.storeFunctor(continuator);

    double threshold = parser.createParam((double)0.1, "threshold", "Threshold: temperature threshold stopping criteria", 't', section).value(); // t
    double alpha = parser.createParam((double)0.1, "alpha", "Alpha: temperature dicrease rate", 'a', section).value(); // a

    moCoolingSchedule* cooling_schedule = new moGeometricCoolingSchedule(threshold, alpha);
    state.storeFunctor(cooling_schedule);

    // stopping criteria
    // ... and creates the parameter letters: C E g G s T

    eoContinue< EOT >& monitoring_continue = do_make_continue(parser, state, eval);

    // output

    eoCheckPoint< EOT >& checkpoint = do_make_checkpoint(parser, state, eval, monitoring_continue);

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
    // CMASA algorithm configuration
    //-----------------------------------------------------------------------------

    //doAlgo< doUniform< EOT > >* algo = new doCMASA< doUniform< EOT > >
    doAlgo< doNormal< EOT > >* algo = new doCMASA< doNormal< EOT > >
    	(*selector, *estimator, *selectone, *modifier, *sampler,
    	 checkpoint, eval, *continuator, *cooling_schedule,
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
