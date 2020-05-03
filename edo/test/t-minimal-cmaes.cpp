#include <eo>
#include <edo>
#include <es.h>
#include <do/make_pop.h>
#include <do/make_run.h>
#include <do/make_continue.h>
#include <do/make_checkpoint.h>

using R = eoReal<eoMinimizingFitness>;
using CMA = edoNormalAdaptive<R>;

R::FitnessType sphere(const R& sol) {
    double sum = 0;
    for(auto x : sol) { sum += x * x; }
    return sum;
}

int main(int argc, char** argv) {
    eoParser parser(argc, argv);
    eoState state;

    size_t dim = parser.createParam<size_t>(10,
            "dimension", "Dimension", 'd',
            "Problem").value();

    size_t max_eval = parser.getORcreateParam<size_t>(100 * dim,
            "maxEval", "Maximum number of evaluations", 'E',
            "Stopping criterion").value();

    edoNormalAdaptive<R> gaussian(dim);

    auto& obj_func = state.pack< eoEvalFuncPtr<R> >(sphere);
    auto& eval     = state.pack< eoEvalCounterThrowException<R> >(obj_func, max_eval);
    auto& pop_eval = state.pack< eoPopLoopEval<R> >(eval);

    auto& gen  = state.pack< eoUniformGenerator<R::AtomType> >(-5, 5);
    auto& init = state.pack< eoInitFixedLength<R> >(dim, gen);
    auto& pop = do_make_pop(parser, state, init);
    pop_eval(pop,pop);

    auto& eo_continue  = do_make_continue(  parser, state, eval);
    auto& pop_continue = do_make_checkpoint(parser, state, eval, eo_continue);
    auto& best = state.pack< eoBestIndividualStat<R> >();
    pop_continue.add( best );
    auto& distrib_continue = state.pack< edoContAdaptiveFinite<CMA> >();

    auto& selector  = state.pack< eoRankMuSelect<R> >(dim/2);
    auto& estimator = state.pack< edoEstimatorNormalAdaptive<R> >(gaussian);
    auto& bounder   = state.pack< edoBounderRng<R> >(R(dim, -5), R(dim, 5), gen);
    auto& sampler   = state.pack< edoSamplerNormalAdaptive<R> >(bounder);
    auto& replacor  = state.pack< eoCommaReplacement<R> >();

    make_verbose(parser);
    make_help(parser);

    auto& algo = state.pack< edoAlgoAdaptive<CMA> >(
         gaussian , pop_eval, selector,
         estimator, sampler , replacor,
         pop_continue, distrib_continue);

    try {
        algo(pop);
    } catch (eoMaxEvalException& e) {
        eo::log << eo::progress << "STOP" << std::endl;
    }

    std::cout << best.value() << std::endl;
    return 0;
}
