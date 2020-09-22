#include <iostream>
#include <cstdlib>
#include <string>

#include <eo>
#include <ga.h>
#include <utils/checkpointing>
#include <eoInt.h>
#include <problems/eval/eoEvalIOH.h>

#include <IOHprofiler_ecdf_logger.h>
#include <f_w_model_one_max.hpp>

// using Particle = eoRealParticle<eoMaximizingFitness>;
using Ints = eoInt<eoMinimizingFitnessT<int>, size_t>;
using Bits = eoBit<eoMinimizingFitnessT<int>, int>;

// by enumerating candidate operators and their parameters.
eoAlgoFoundryFastGA<Bits>& make_foundry(
        eoFunctorStore& store,
        eoInit<Bits>& init,
        eoEvalFunc<Bits>& eval_onemax,
        const size_t max_evals,
        const size_t generations
    )
{
    auto& foundry = store.pack< eoAlgoFoundryFastGA<Bits> >(init, eval_onemax, max_evals /*, max_restarts = max */);

    /***** Continuators ****/
    foundry.continuators.add< eoGenContinue<Bits> >(generations);
    // for(size_t i=1; i<10; i++) {
    //     foundry.continuators.add< eoGenContinue<Bits> >(i);
    // }
    // for(size_t i=10; i < 100; i+=2 ) {
    //     foundry.continuators.add< eoSteadyFitContinue<Bits> >(10,i);
    // }

    for(double i=0.1; i<1.0; i+=0.1) {
        foundry.crossover_rates.add<double>(i);
        foundry.mutation_rates.add<double>(i);
    }

    for(size_t i=5; i<100; i+=10) {
        foundry.pop_sizes.add<size_t>(i);
    }

    /***** Crossovers ****/
    for(double i=0.1; i<0.9; i+=0.1) {
        foundry.crossovers.add< eoUBitXover<Bits> >(i); // preference over 1
    }
    for(size_t i=1; i < 11; i+=1) {
        foundry.crossovers.add< eoNPtsBitXover<Bits> >(i); // nb of points
    }
    foundry.crossovers.add< eo1PtBitXover<Bits> >();

    /***** Mutations ****/
    double p = 1.0; // Probability of flipping eath bit.
    foundry.mutations.add< eoUniformBitMutation<Bits> >(p); // proba of flipping k bits, k drawn in uniform distrib
    foundry.mutations.add< eoStandardBitMutation<Bits> >(p); // proba of flipping k bits, k drawn in binomial distrib
    foundry.mutations.add< eoConditionalBitMutation<Bits> >(p); // proba of flipping k bits, k drawn in binomial distrib, minus zero
    foundry.mutations.add< eoShiftedBitMutation<Bits> >(p); // proba of flipping k bits, k drawn in binomial distrib, changing zeros to one
    foundry.mutations.add< eoNormalBitMutation<Bits> >(p); // proba of flipping k bits, k drawn in normal distrib
    foundry.mutations.add< eoFastBitMutation<Bits> >(p); // proba of flipping k bits, k drawn in powerlaw distrib
    for(size_t i=1; i < 11; i+=1) {
        foundry.mutations.add< eoDetSingleBitFlip<Bits> >(i); // mutate k bits without duplicates
    }

    /***** Selectors *****/
    foundry.selectors.add< eoRandomSelect<Bits> >();
    foundry.selectors.add< eoSequentialSelect<Bits> >();
    foundry.selectors.add< eoProportionalSelect<Bits> >();
    for(size_t i=2; i < 10; i+=1) { // Tournament size.
        foundry.selectors.add< eoDetTournamentSelect<Bits> >(i);
    }
    for(double i=0.51; i<0.91; i+=0.1) { // Tournament size as perc of pop.
        foundry.selectors.add< eoStochTournamentSelect<Bits> >(i);
    }

    /***** Replacements ****/
    foundry.replacements.add< eoPlusReplacement<Bits> >();
    foundry.replacements.add< eoCommaReplacement<Bits> >();
    foundry.replacements.add< eoSSGAWorseReplacement<Bits> >();
    for(double i=0.51; i<0.91; i+=0.1) {
        foundry.replacements.add< eoSSGAStochTournamentReplacement<Bits> >(i);
    }
    for(size_t i=2; i < 10; i+=1) {
        foundry.replacements.add< eoSSGADetTournamentReplacement<Bits> >(i);
    }

    return foundry;
}

Bits::Fitness fake_func(const Bits&) { return 0; }

void print_param_range(const eoParam& param, const size_t slot_size, std::ostream& out = std::cout)
{
    out << param.longName()
        << "\t\"--" << param.longName() << "=\""
        << "\ti"
        << "\t(0," << slot_size << ")"
        << std::endl;
}

int main(int argc, char* argv[])
{
    /***** Global parameters. *****/
    enum { NO_ERROR = 0, ERROR_USAGE = 100 };

    eoFunctorStore store;

    eoParser parser(argc, argv, "FastGA interface for iRace");

    const size_t dimension = parser.getORcreateParam<size_t>(1000,
            "dimension", "Dimension size",
            'd', "Problem").value();

    const size_t max_evals = parser.getORcreateParam<size_t>(2 * dimension,
            "max-evals", "Maximum number of evaluations",
            'e', "Stopping criterion").value();

    const size_t buckets = parser.getORcreateParam<size_t>(100,
            "buckets", "Number of buckets for discretizing the ECDF",
            'b', "Performance estimation").value();

    uint32_t seed =
        parser.getORcreateParam<uint32_t>(0,
            "seed", "Random number seed (0 = epoch)",
            'S').value();
    if(seed == 0) {
        seed = time(0);
    }
    // rng is a global
    rng.reseed(seed);

    auto instance_p = parser.getORcreateParam<size_t>(1,
            "instance", "Instance ID",
            'i', "Problem", /*required=*/true);
    const size_t instance = instance_p.value();

    auto continuator_p = parser.getORcreateParam<size_t>(0,
            "continuator", "",
            'o', "Evolution Engine", /*required=*/true);
    const size_t continuator = continuator_p.value();

    auto crossover_rate_p = parser.getORcreateParam<size_t>(0,
            "crossover-rate", "",
            'C', "Evolution Engine", /*required=*/true);
    const size_t crossover_rate = crossover_rate_p.value();

    auto crossover_p = parser.getORcreateParam<size_t>(0,
            "crossover", "",
            'c', "Evolution Engine", /*required=*/true);
    const size_t crossover = crossover_p.value();

    auto mutation_rate_p = parser.getORcreateParam<size_t>(0,
            "mutation-rate", "",
            'M', "Evolution Engine", /*required=*/true);
    const size_t mutation_rate = mutation_rate_p.value();

    auto mutation_p = parser.getORcreateParam<size_t>(0,
            "mutation", "",
            'm', "Evolution Engine", /*required=*/true);
    const size_t mutation = mutation_p.value();

    auto selector_p = parser.getORcreateParam<size_t>(0,
            "selector", "",
            's', "Evolution Engine", /*required=*/true);
    const size_t selector = selector_p.value();

    auto pop_size_p = parser.getORcreateParam<size_t>(0,
            "pop-size", "",
            'P', "Evolution Engine", /*required=*/true);
    const size_t pop_size = pop_size_p.value();

    auto replacement_p = parser.getORcreateParam<size_t>(0,
            "replacement", "",
            'r', "Evolution Engine", /*required=*/true);
    const size_t replacement = replacement_p.value();

    // Help + Verbose routines
    make_verbose(parser);
    make_help(parser, /*exit_after*/false, std::clog);

    if(parser.userNeedsHelp()) {

        // Fake operators, just to be able to call make_foundry
        // to get the configured operators slots.
        eoEvalFuncPtr<Bits> fake_eval(fake_func);
        eoUniformGenerator<int> fake_gen(0, 1);
        eoInitFixedLength<Bits> fake_init(/*bitstring size=*/1, fake_gen);
        auto fake_foundry = make_foundry(store, fake_init, fake_eval, max_evals, /*generations=*/ 1);

        size_t n = fake_foundry.continuators.size()
                 * fake_foundry.crossovers.size()
                 * fake_foundry.mutations.size()
                 * fake_foundry.selectors.size()
                 * fake_foundry.replacements.size();
        std::clog << n << " possible algorithms instances." << std::endl;

        std::clog << "Ranges of required parameters (redirect the stdout in a file to use it with iRace): " << std::endl;

        std::cout << "# name\tswitch\ttype\trange" << std::endl;
        print_param_range(      instance_p, 18, std::cout);
        print_param_range(   continuator_p, fake_foundry.continuators   .size(), std::cout);
        print_param_range(     crossover_p, fake_foundry.crossovers     .size(), std::cout);
        print_param_range(crossover_rate_p, fake_foundry.crossover_rates.size(), std::cout);
        print_param_range(      mutation_p, fake_foundry.mutations      .size(), std::cout);
        print_param_range( mutation_rate_p, fake_foundry.mutation_rates .size(), std::cout);
        print_param_range(      selector_p, fake_foundry.selectors      .size(), std::cout);
        print_param_range(      pop_size_p, fake_foundry.pop_sizes      .size(), std::cout);
        print_param_range(   replacement_p, fake_foundry.replacements   .size(), std::cout);

        // std::ofstream irace_param("fastga.params");
        // irace_param << "# name\tswitch\ttype\tvalues" << std::endl;

        exit(NO_ERROR);
    }

    const size_t generations = static_cast<size_t>(std::floor(
                static_cast<double>(max_evals) / static_cast<double>(pop_size)));

    Ints encoded_algo(8);
    encoded_algo[0] = continuator;
    encoded_algo[1] = crossover_rate;
    encoded_algo[2] = crossover;
    encoded_algo[3] = mutation_rate;
    encoded_algo[4] = mutation;
    encoded_algo[5] = selector;
    encoded_algo[6] = pop_size;
    encoded_algo[7] = replacement;

    /***** IOH logger *****/
    IOHprofiler_RangeLinear<size_t> target_range(0, dimension, buckets);
    IOHprofiler_RangeLinear<size_t> budget_range(0, max_evals, buckets);
    IOHprofiler_ecdf_logger<int, size_t, size_t> logger(target_range, budget_range);

    logger.set_complete_flag(true);
    logger.set_interval(0);
    logger.activate_logger();

    /***** IOH problem *****/
    double w_model_suite_dummy_para = 0;
    int w_model_suite_epitasis_para = 0;
    int w_model_suite_neutrality_para = 0;
    int w_model_suite_ruggedness_para = 0;

    W_Model_OneMax w_model_om;
    std::string problem_name = "OneMax";
    problem_name = problem_name
                    + "_D" + std::to_string((int)(w_model_suite_dummy_para * dimension))
                    + "_E" + std::to_string(w_model_suite_epitasis_para)
                    + "_N" + std::to_string(w_model_suite_neutrality_para)
                    + "_R" + std::to_string(w_model_suite_ruggedness_para);

    /// This must be called to configure the w-model to be tested.
    w_model_om.set_w_setting(w_model_suite_dummy_para,w_model_suite_epitasis_para,
                                    w_model_suite_neutrality_para,w_model_suite_ruggedness_para);

    /// Set problem_name based on the configuration.
    w_model_om.IOHprofiler_set_problem_name(problem_name);

    /// Set problem_id as 1
    w_model_om.IOHprofiler_set_problem_id(1);
    w_model_om.IOHprofiler_set_instance_id(instance);

    /// Set dimension.
    w_model_om.IOHprofiler_set_number_of_variables(dimension);

    /***** Bindings *****/
    logger.track_problem(w_model_om);

    eoEvalIOHproblem<Bits> onemax_eval(w_model_om, logger);
    eoPopLoopEval<Bits> pop_onemax(onemax_eval);

    /***** Instanciate and run the algo *****/

    eoUniformGenerator<int> ugen(0, 1);
    eoInitFixedLength<Bits> onemax_init(/*bitstring size=*/dimension, ugen);
    auto& foundry = make_foundry(store, onemax_init, onemax_eval, max_evals, generations);

    std::clog << "Encoded algorithm:" << std::endl;
    foundry.select(encoded_algo);
    std::clog << "\tcontinuator:\t"    << foundry.continuator   ().className() << std::endl;
    std::clog << "\tcrossover:\t"      << foundry.crossover     ().className() << std::endl;
    std::clog << "\tcrossover_rate:\t" << foundry.crossover_rate()             << std::endl;
    std::clog << "\tmutation:\t"       << foundry.mutation      ().className() << std::endl;
    std::clog << "\tmutation_rate:\t"  << foundry.mutation_rate ()             << std::endl;
    std::clog << "\tselector:\t"       << foundry.selector      ().className() << std::endl;
    std::clog << "\tpop_size:\t"       << foundry.pop_size      ()             << std::endl;
    std::clog << "\treplacement:\t"    << foundry.replacement   ().className() << std::endl;

    // Evaluation of a forged encoded_algo on the sub-problem
    eoEvalFoundryFastGA<Ints, Bits> eval_foundry(
            foundry, onemax_init, pop_onemax, /*penalization=*/ 0);

    // Actually instanciate and run the algorithm.
    eval_foundry(encoded_algo);

    /***** IOH perf stats *****/
    IOHprofiler_ecdf_sum ecdf_sum;
    // iRace expects minimization
    long perf = ecdf_sum(logger.data());

    // Output
    std::cout << -1 * perf << std::endl;

    assert(0 < perf and perf <= buckets*buckets);
}
