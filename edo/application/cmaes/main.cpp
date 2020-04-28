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
//#include <mo>

#include <eoEvalCounterThrowException.h>

#include <do/make_pop.h>
#include <do/make_run.h>
#include <do/make_continue.h>
#include <do/make_checkpoint.h>

#include <edo>

#include "Rosenbrock.h"
#include "Sphere.h"


typedef eoReal<eoMinimizingFitness> RealVec;
typedef edoNormalAdaptive< RealVec > Distrib;


int main(int ac, char** av)
{
    eoParser parser(ac, av);

    eoState state;
    // Letters used by the following declarations:
    unsigned long max_eval = parser.getORcreateParam((unsigned long)0, "maxEval", "Maximum number of evaluations (0 = none)", 'E', "Stopping criterion").value(); // E
    unsigned int dim = parser.createParam((unsigned int)10, "dimension-size", "Dimension size", 'd', "Problem").value(); // d

    double mu = dim / 2;

    edoNormalAdaptive<RealVec> distribution(dim);

    eoSelect< RealVec >* selector = new eoRankMuSelect< RealVec >( mu );
    state.storeFunctor(selector);

    edoEstimator< Distrib >* estimator = new edoEstimatorNormalAdaptive<RealVec>( distribution );
    state.storeFunctor(estimator);

    eoEvalFunc< RealVec >* plainEval = new Rosenbrock< RealVec >();
    state.storeFunctor(plainEval);

    eoEvalCounterThrowException< RealVec > eval(*plainEval, max_eval);

    eoRndGenerator< double >* gen = new eoUniformGenerator< double >(-5, 5);
    state.storeFunctor(gen);


    eoInitFixedLength< RealVec >* init = new eoInitFixedLength< RealVec >( dim, *gen );
    state.storeFunctor(init);


    // (1) Population init and sampler
    // Generation of population from do_make_pop (creates parameters, manages persistance and so on...)
    // ... and creates the parameters: L P r S
    // this first sampler creates a uniform distribution independently from our distribution (it does not use edoUniform).
    eoPop< RealVec >& pop = do_make_pop(parser, state, *init);

    // (2) First evaluation before starting the research algorithm
    ::apply(eval, pop);

    // Prepare bounder class to set bounds of sampling.
    // This is used by edoSampler.
    edoBounder< RealVec >* bounder = 
        new edoBounderRng< RealVec >( RealVec(dim, -5), RealVec(dim, 5), *gen); // FIXME do not use hard-coded bounds
    state.storeFunctor(bounder);

    // Prepare sampler class with a specific distribution
    edoSampler< Distrib >* sampler = new edoSamplerNormalAdaptive< RealVec >( *bounder );
    state.storeFunctor(sampler);

    // stopping criteria
    // ... and creates the parameter letters: C E g G s T
    eoContinue< RealVec >& eo_continue = do_make_continue(parser, state, eval);

    // population output
    eoCheckPoint< RealVec >& pop_continue = do_make_checkpoint(parser, state, eval, eo_continue);
    // keep the best solution found so far in an eoStat
    // thus, if the population's best individual fitness decreases during the search, we could
    // still keep the best found since the beginning, while avoiding the bias of elitism on the sample
    eoBestIndividualStat<RealVec> best_indiv;
    pop_continue.add( best_indiv );

    // distribution output
    edoDummyContinue< Distrib >* dummy_continue = new edoDummyContinue< Distrib >();
    state.storeFunctor(dummy_continue);

    edoCheckPoint< Distrib >* distribution_continue = new edoCheckPoint< Distrib >( *dummy_continue );
    state.storeFunctor(distribution_continue);

    eoReplacement< RealVec >* replacor = new eoCommaReplacement< RealVec >();
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

    eoPopLoopEval<RealVec> popEval( eval );

    // CMA-ES algorithm configuration
    edoAlgo< Distrib >* algo = new edoAlgoAdaptive< Distrib >
        (distribution, popEval, *selector, *estimator, *sampler, *replacor,
         pop_continue, *distribution_continue );

    // Use the best solution of the random first pop to start the search
    // That is, center the distribution's mean on it.
    distribution.mean( pop.best_element() );

    // Beginning of the algorithm call
    try {
        eo::log << eo::progress << "Best solution after random init: " << pop.best_element().fitness() << std::endl;
        do_run(*algo, pop);

    } catch (eoMaxEvalException& e) {
            eo::log << eo::warnings << "warning: " << e.what() << std::endl;
    }

    // use the stat instead of the pop, to get the best solution of the whole search
    std::cout << best_indiv.value() << std::endl;

    return 0;
}
