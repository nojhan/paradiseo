#include <eo>
#include <mo>

#include <eoEvalFuncCounterBounder.h>

#include <do/make_pop.h>
#include <do/make_run.h>
#include <do/make_continue.h>
#include <do/make_checkpoint.h>

#include <edo>

#include "Rosenbrock.h"
#include "Sphere.h"


typedef eoReal<eoMinimizingFitness> EOT;
typedef edoNormalMulti< EOT > Distrib;


int main(int ac, char** av)
{
    eoParser parser(ac, av);

    // Letters used by the following declarations:
    // a d i p t

    std::string	section("Algorithm parameters");

    // FIXME: default value to check
    double initial_temperature = parser.createParam((double)10e5, "temperature", "Initial temperature", 'i', section).value(); // i

    eoState state;


    //-----------------------------------------------------------------------------
    // Instantiate all needed parameters for EDASA algorithm
    //-----------------------------------------------------------------------------

    double selection_rate = parser.createParam((double)0.5, "selection_rate", "Selection Rate", 'R', section).value(); // R

    eoSelect< EOT >* selector = new eoDetSelect< EOT >( selection_rate );
    state.storeFunctor(selector);

    edoEstimator< Distrib >* estimator =	new edoEstimatorNormalMulti< EOT >();
    state.storeFunctor(estimator);

    eoSelectOne< EOT >* selectone = new eoDetTournamentSelect< EOT >( 2 );
    state.storeFunctor(selectone);

    edoModifierMass< Distrib >* modifier = new edoNormalMultiCenter< EOT >();
    state.storeFunctor(modifier);

    eoEvalFunc< EOT >* plainEval = new Rosenbrock< EOT >();
    state.storeFunctor(plainEval);

    unsigned long max_eval = parser.getORcreateParam((unsigned long)0, "maxEval", "Maximum number of evaluations (0 = none)", 'E', "Stopping criterion").value(); // E
    eoEvalFuncCounterBounder< EOT > eval(*plainEval, max_eval);

    eoRndGenerator< double >* gen = new eoUniformGenerator< double >(-5, 5);
    state.storeFunctor(gen);


    unsigned int dimension_size = parser.createParam((unsigned int)10, "dimension-size", "Dimension size", 'd', section).value(); // d

    eoInitFixedLength< EOT >* init = new eoInitFixedLength< EOT >( dimension_size, *gen );
    state.storeFunctor(init);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // (1) Population init and sampler
    //-----------------------------------------------------------------------------

    // Generation of population from do_make_pop (creates parameters, manages persistance and so on...)
    // ... and creates the parameters: L P r S

    // this first sampler creates a uniform distribution independently from our distribution (it does not use doUniform).

    eoPop< EOT >& pop = do_make_pop(parser, state, *init);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // (2) First evaluation before starting the research algorithm
    //-----------------------------------------------------------------------------

    apply(eval, pop);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // Prepare bounder class to set bounds of sampling.
    // This is used by doSampler.
    //-----------------------------------------------------------------------------

    edoBounder< EOT >* bounder = new edoBounderRng< EOT >(EOT(pop[0].size(), -5),
							EOT(pop[0].size(), 5),
							*gen);
    state.storeFunctor(bounder);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // Prepare sampler class with a specific distribution
    //-----------------------------------------------------------------------------

    edoSampler< Distrib >* sampler = new edoSamplerNormalMulti< EOT >( *bounder );
    state.storeFunctor(sampler);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // Metropolis sample parameters
    //-----------------------------------------------------------------------------

    unsigned int popSize = parser.getORcreateParam((unsigned int)20, "popSize", "Population Size", 'P', "Evolution Engine").value();

    moContinuator< moDummyNeighbor<EOT> >* sa_continue = new moIterContinuator< moDummyNeighbor<EOT> >( popSize );
    state.storeFunctor(sa_continue);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // SA parameters
    //-----------------------------------------------------------------------------

    double threshold_temperature = parser.createParam((double)0.1, "threshold", "Minimal temperature at which stop", 't', section).value(); // t
    double alpha = parser.createParam((double)0.1, "alpha", "Temperature decrease rate", 'a', section).value(); // a

    moCoolingSchedule<EOT>* cooling_schedule = new moSimpleCoolingSchedule<EOT>(initial_temperature, alpha, 0, threshold_temperature);
    state.storeFunctor(cooling_schedule);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // stopping criteria
    // ... and creates the parameter letters: C E g G s T
    //-----------------------------------------------------------------------------

    eoContinue< EOT >& eo_continue = do_make_continue(parser, state, eval);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // population output
    //-----------------------------------------------------------------------------

    eoCheckPoint< EOT >& pop_continue = do_make_checkpoint(parser, state, eval, eo_continue);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // distribution output
    //-----------------------------------------------------------------------------

    edoDummyContinue< Distrib >* dummy_continue = new edoDummyContinue< Distrib >();
    state.storeFunctor(dummy_continue);

    edoCheckPoint< Distrib >* distribution_continue = new edoCheckPoint< Distrib >( *dummy_continue );
    state.storeFunctor(distribution_continue);

    //-----------------------------------------------------------------------------


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
    // Some stuff to display helper when we are using -h option
    //-----------------------------------------------------------------------------

    if (parser.userNeedsHelp())
	{
	    parser.printHelp(std::cout);
	    exit(1);
	}

    // Help + Verbose routines

    make_verbose(parser);
    make_help(parser);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // population output (after helper)
    //
    // FIXME: theses objects are instanciate there in order to avoid a folder
    // removing as edoFileSnapshot does within ctor.
    //-----------------------------------------------------------------------------

    edoPopStat< EOT >* popStat = new edoPopStat<EOT>;
    state.storeFunctor(popStat);
    pop_continue.add(*popStat);

    edoFileSnapshot* fileSnapshot = new edoFileSnapshot("EDASA_ResPop");
    state.storeFunctor(fileSnapshot);
    fileSnapshot->add(*popStat);
    pop_continue.add(*fileSnapshot);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // distribution output (after helper)
    //-----------------------------------------------------------------------------

    edoDistribStat< Distrib >* distrib_stat = new edoStatNormalMulti< EOT >();
    state.storeFunctor(distrib_stat);

    distribution_continue->add( *distrib_stat );

    // eoMonitor* stdout_monitor = new eoStdoutMonitor();
    // state.storeFunctor(stdout_monitor);
    // stdout_monitor->add(*distrib_stat);
    // distribution_continue->add( *stdout_monitor );

    eoFileMonitor* file_monitor = new eoFileMonitor("eda_sa_distribution_bounds.txt");
    state.storeFunctor(file_monitor);
    file_monitor->add(*distrib_stat);
    distribution_continue->add( *file_monitor );

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // EDASA algorithm configuration
    //-----------------------------------------------------------------------------

    edoAlgo< Distrib >* algo = new edoEDASA< Distrib >
    	(*selector, *estimator, *selectone, *modifier, *sampler,
	 pop_continue, *distribution_continue,
	 eval, *sa_continue, *cooling_schedule,
    	 initial_temperature, *replacor);

    //-----------------------------------------------------------------------------


    //-----------------------------------------------------------------------------
    // Beginning of the algorithm call
    //-----------------------------------------------------------------------------

    try
	{
	    do_run(*algo, pop);
	}
    catch (eoEvalFuncCounterBounderException& e)
    	{
    	    eo::log << eo::warnings << "warning: " << e.what() << std::endl;
    	}
    catch (std::exception& e)
	{
	    eo::log << eo::errors << "error: " << e.what() << std::endl;
	    exit(EXIT_FAILURE);
	}

    //-----------------------------------------------------------------------------

    return 0;
}
