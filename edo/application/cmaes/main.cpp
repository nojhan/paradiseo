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

#include <eoEvalFuncCounterBounder.h>

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

    // Letters used by the following declarations:
    // a d i p t

    std::string    section("Algorithm parameters");

    eoState state;

    // Instantiate all needed parameters for EDA algorithm
    //double selection_rate = parser.createParam((double)0.5, "selection_rate", "Selection Rate", 'R', section).value(); // R

    unsigned long max_eval = parser.getORcreateParam((unsigned long)0, "maxEval", "Maximum number of evaluations (0 = none)", 'E', "Stopping criterion").value(); // E

    unsigned int dim = parser.createParam((unsigned int)10, "dimension-size", "Dimension size", 'd', section).value(); // d


    double mu = dim / 2;


    edoNormalAdaptive<RealVec> distribution(dim);

    eoSelect< RealVec >* selector = new eoRankMuSelect< RealVec >( mu );
    state.storeFunctor(selector);

    edoEstimator< Distrib >* estimator = new edoEstimatorNormalAdaptive<RealVec>( distribution );
    state.storeFunctor(estimator);

    eoEvalFunc< RealVec >* plainEval = new Rosenbrock< RealVec >();
    state.storeFunctor(plainEval);

    eoEvalFuncCounterBounder< RealVec > eval(*plainEval, max_eval);

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
    apply(eval, pop);

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

    // distribution output
    edoDummyContinue< Distrib >* dummy_continue = new edoDummyContinue< Distrib >();
    state.storeFunctor(dummy_continue);

    edoCheckPoint< Distrib >* distribution_continue = new edoCheckPoint< Distrib >( *dummy_continue );
    state.storeFunctor(distribution_continue);

    // eoEPRemplacement causes the using of the current and previous
    // sample for sampling.
    eoReplacement< RealVec >* replacor = new eoEPReplacement< RealVec >(pop.size());
    state.storeFunctor(replacor);

    // Some stuff to display helper when we are using -h option
    if (parser.userNeedsHelp())
    {
        parser.printHelp(std::cout);
        exit(1);
    }

    // Help + Verbose routines
    make_verbose(parser);
    make_help(parser);

    eoPopLoopEval<RealVec> popEval( eval );

    // EDA algorithm configuration
    edoAlgo< Distrib >* algo = new edoAlgoAdaptive< Distrib >
        (distribution, popEval, *selector, *estimator, *sampler, *replacor,
         pop_continue, *distribution_continue );


    // Beginning of the algorithm call
    try {
        do_run(*algo, pop);

    } catch (eoEvalFuncCounterBounderException& e) {
            eo::log << eo::warnings << "warning: " << e.what() << std::endl;

    } catch (std::exception& e) {
        eo::log << eo::errors << "error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
