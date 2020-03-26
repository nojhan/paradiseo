/*
The Evolving Distribution Objects framework (EDO) is a template-based,
ANSI-C++ evolutionary computation library which helps you to write your
own estimation of distribution algorithms.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (C) 2010 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
    Caner Candan <caner.candan@thalesgroup.com>
*/

#include <eo>
// #include <mo>

#include <eoEvalCounterThrowException.h>

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

    std::string    section("Algorithm parameters");

    eoState state;

    // Instantiate all needed parameters for EDA algorithm
    double selection_rate = parser.createParam((double)0.5, "selection_rate", "Selection Rate", 'R', section).value(); // R

    eoSelect< EOT >* selector = new eoDetSelect< EOT >( selection_rate );
    state.storeFunctor(selector);

    edoEstimator< Distrib >* estimator =    new edoEstimatorNormalMulti< EOT >();
    state.storeFunctor(estimator);

    eoEvalFunc< EOT >* plainEval = new Rosenbrock< EOT >();
    state.storeFunctor(plainEval);

    unsigned long max_eval = parser.getORcreateParam((unsigned long)0, "maxEval", "Maximum number of evaluations (0 = none)", 'E', "Stopping criterion").value(); // E
    eoEvalCounterThrowException< EOT > eval(*plainEval, max_eval);

    eoRndGenerator< double >* gen = new eoUniformGenerator< double >(-5, 5);
    state.storeFunctor(gen);

    unsigned int dimension_size = parser.createParam((unsigned int)10, "dimension-size", "Dimension size", 'd', section).value(); // d

    eoInitFixedLength< EOT >* init = new eoInitFixedLength< EOT >( dimension_size, *gen );
    state.storeFunctor(init);


    // (1) Population init and sampler
    // Generation of population from do_make_pop (creates parameters, manages persistance and so on...)
    // ... and creates the parameters: L P r S
    // this first sampler creates a uniform distribution independently from our distribution (it does not use edoUniform).
    eoPop< EOT >& pop = do_make_pop(parser, state, *init);

    // (2) First evaluation before starting the research algorithm
    apply(eval, pop);

    // Prepare bounder class to set bounds of sampling.
    // This is used by edoSampler.
    edoBounder< EOT >* bounder = 
        new edoBounderRng< EOT >( EOT(dimension_size, -5), EOT(dimension_size, 5), *gen); // FIXME do not use hard-coded bounds
    state.storeFunctor(bounder);

    // Prepare sampler class with a specific distribution
    edoSampler< Distrib >* sampler = new edoSamplerNormalMulti< EOT >( *bounder );
    state.storeFunctor(sampler);
    
    // stopping criteria
    // ... and creates the parameter letters: C E g G s T
    eoContinue< EOT >& eo_continue = do_make_continue(parser, state, eval);
    
    // population output
    eoCheckPoint< EOT >& pop_continue = do_make_checkpoint(parser, state, eval, eo_continue);
    
    // distribution output
    edoDummyContinue< Distrib >* dummy_continue = new edoDummyContinue< Distrib >();
    state.storeFunctor(dummy_continue);

    edoCheckPoint< Distrib >* distribution_continue = new edoCheckPoint< Distrib >( *dummy_continue );
    state.storeFunctor(distribution_continue);

    // eoEPRemplacement causes the using of the current and previous
    // sample for sampling.
    eoReplacement< EOT >* replacor = new eoEPReplacement< EOT >(pop.size());
    state.storeFunctor(replacor);

    // Help + Verbose routines
    make_verbose(parser);
    make_help(parser);

    // Some stuff to display helper when we are using -h option
    if (parser.userNeedsHelp())
    {
        parser.printHelp(std::cout);
        exit(1);
    }

    // population output (after helper)
    //
    // FIXME: theses objects are instanciated there in order to avoid a folder
    // removing as edoFileSnapshot does within ctor.
    edoPopStat< EOT >* popStat = new edoPopStat<EOT>;
    state.storeFunctor(popStat);
    pop_continue.add(*popStat);

    edoFileSnapshot* fileSnapshot = new edoFileSnapshot("EDA_ResPop");
    state.storeFunctor(fileSnapshot);
    fileSnapshot->add(*popStat);
    pop_continue.add(*fileSnapshot);

    // distribution output (after helper)
    edoDistribStat< Distrib >* distrib_stat = new edoStatNormalMulti< EOT >();
    state.storeFunctor(distrib_stat);

    distribution_continue->add( *distrib_stat );

    // eoMonitor* stdout_monitor = new eoStdoutMonitor();
    // state.storeFunctor(stdout_monitor);
    // stdout_monitor->add(*distrib_stat);
    // distribution_continue->add( *stdout_monitor );

    eoFileMonitor* file_monitor = new eoFileMonitor("eda_distribution_bounds.txt");
    state.storeFunctor(file_monitor);
    file_monitor->add(*distrib_stat);
    distribution_continue->add( *file_monitor );

    eoPopLoopEval<EOT> popEval( eval );

    // EDA algorithm configuration
    edoAlgo< Distrib >* algo = new edoAlgoStateless< Distrib >
        (popEval, *selector, *estimator, *sampler, *replacor,
         pop_continue, *distribution_continue );

    // Beginning of the algorithm call
    try {
        do_run(*algo, pop);

    } catch (eoMaxEvalException& e) {
            eo::log << eo::warnings << "warning: " << e.what() << std::endl;

    } catch (std::exception& e) {
        eo::log << eo::errors << "error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
