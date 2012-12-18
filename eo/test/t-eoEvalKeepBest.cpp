#include <iostream>

#include <es/make_real.h>
#include <apply.h>
#include <eoEvalKeepBest.h>
#include "real_value.h"

using namespace std;

int main(int argc, char* argv[])
{

    typedef eoReal<eoMinimizingFitness> EOT;

    eoParser parser(argc, argv);  // for user-parameter reading
    eoState state;    // keeps all things allocated


    /*********************************************
     * problem or representation dependent stuff *
     *********************************************/

    // The evaluation fn - encapsulated into an eval counter for output
    eoEvalFuncPtr<EOT, double, const std::vector<double>&>
        main_eval( real_value ); // use a function defined in real_value.h

    // wrap the evaluation function in a call counter
    eoEvalFuncCounter<EOT> eval_counter(main_eval);

    // the genotype - through a genotype initializer
    eoRealInitBounded<EOT>& init = make_genotype(parser, state, EOT());

    // Build the variation operator (any seq/prop construct)
    eoGenOp<EOT>& op = make_op(parser, state, init);


    /*********************************************
     * Now the representation-independent things *
     *********************************************/


    // initialize the population - and evaluate
    // yes, this is representation indepedent once you have an eoInit
    eoPop<EOT>& pop   = make_pop(parser, state, init);

    // stopping criteria
    eoContinue<EOT> & term = make_continue(parser, state, eval_counter);

    // things that are called at each generation
    eoCheckPoint<EOT> & checkpoint = make_checkpoint(parser, state, eval_counter, term);

    // wrap the evaluator in another one that will keep the best individual
    // evaluated so far
    eoEvalKeepBest<EOT> eval_keep( eval_counter );

    // algorithm
    eoAlgo<EOT>& ea = make_algo_scalar(parser, state, eval_keep, checkpoint, op);


    /***************************************
     * Now, call functors and DO something *
     ***************************************/

    // to be called AFTER all parameters have been read!
    make_help(parser);

    // evaluate intial population AFTER help and status in case it takes time
    apply<EOT>(eval_keep, pop);

    std::clog << "Best individual after initialization and " << eval_counter.value() << " evaluations" << std::endl;
    std::cout << eval_keep.best_element() << std::endl;

    ea(pop); // run the ea

    std::cout << "Best individual after search and " << eval_counter.value() << " evaluations" << std::endl;
    // you can also call value(), because it is an eoParam
    std::cout << eval_keep.value() << std::endl;
}
