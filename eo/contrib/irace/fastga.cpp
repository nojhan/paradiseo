#include <iostream>
#include <cstdlib>
#include <string>
#include <eo>
#include <ga.h>
#include <utils/checkpointing>
#include <eoInt.h>
#include <problems/eval/eoEvalIOH.h>

#include <IOHprofiler_ecdf_logger.h>
#include <IOHprofiler_csv_logger.h>
#include <IOHprofiler_observer_combine.h>
#include <f_w_model_one_max.hpp>

// using Particle = eoRealParticle<eoMaximizingFitness>;
using Ints = eoInt<eoMaximizingFitnessT<int>, size_t>;
using Bits = eoBit<eoMaximizingFitnessT<int>, int>;

// by enumerating candidate operators and their parameters.
eoAlgoFoundryFastGA<Bits>& make_foundry(
        eoFunctorStore& store,
        eoInit<Bits>& init,
        eoEvalFunc<Bits>& eval_onemax,
        const size_t max_evals,
        const size_t generations
    )
{
    // FIXME using max_restarts>1 does not allow to honor max evals.
    auto& foundry = store.pack< eoAlgoFoundryFastGA<Bits> >(init, eval_onemax, max_evals, /*max_restarts=*/1);

    /***** Continuators ****/
    foundry.continuators.add< eoGenContinue<Bits> >(generations);
    // for(size_t i=1; i<10; i++) {
    //     foundry.continuators.add< eoGenContinue<Bits> >(i);
    // }
    // for(size_t i=10; i < 100; i+=2 ) {
    //     foundry.continuators.add< eoSteadyFitContinue<Bits> >(10,i);
    // }

    for(double i=0.0; i<1.0; i+=0.2) {
        foundry.crossover_rates.add<double>(i);
        foundry.mutation_rates.add<double>(i);
    }

    /***** Offsprings size *****/
    // for(size_t i=5; i<100; i+=10) {
    //     foundry.offspring_sizes.add<size_t>(i);
    // }

    foundry.offspring_sizes.add<size_t>(0); // 0 = use parents fixed pop size.

    /***** Crossovers ****/
    for(double i=0.1; i<1.0; i+=0.2) {
        foundry.crossovers.add< eoUBitXover<Bits> >(i); // preference over 1
    }
    for(size_t i=1; i < 10; i+=2) {

        foundry.crossovers.add< eoNPtsBitXover<Bits> >(i); // nb of points
    }
    foundry.crossovers.add< eo1PtBitXover<Bits> >();

    /***** Mutations ****/
    double p = 1.0; // Probability of flipping each bit.
    // proba of flipping k bits, k drawn in uniform distrib
    foundry.mutations.add< eoUniformBitMutation<Bits> >(p);
    // proba of flipping k bits, k drawn in binomial distrib
    foundry.mutations.add< eoStandardBitMutation<Bits> >(p);
    // proba of flipping k bits, k drawn in binomial distrib, minus zero
    foundry.mutations.add< eoConditionalBitMutation<Bits> >(p);
    // proba of flipping k bits, k drawn in binomial distrib, changing zeros to one
    foundry.mutations.add< eoShiftedBitMutation<Bits> >(p);
    // proba of flipping k bits, k drawn in normal distrib
    foundry.mutations.add< eoNormalBitMutation<Bits> >(p);
    // proba of flipping k bits, k drawn in powerlaw distrib
    foundry.mutations.add< eoFastBitMutation<Bits> >(p);
    for(size_t i=1; i < 11; i+=2) {
        // mutate k bits without duplicates
        foundry.mutations.add< eoDetSingleBitFlip<Bits> >(i);
    }

    /***** Selectors *****/
    for(eoOperatorFoundry<eoSelectOne<Bits>>& ops :
        {std::ref(foundry.crossover_selectors),
         std::ref(foundry.mutation_selectors) }) {

        ops.add< eoRandomSelect<Bits> >();
        ops.add< eoStochTournamentSelect<Bits> >(0.5);
        ops.add< eoSequentialSelect<Bits> >();
        ops.add< eoProportionalSelect<Bits> >();
        for(size_t i=2; i < 11; i+=4) {
            ops.add< eoDetTournamentSelect<Bits> >(i);
        }
    }

    foundry.aftercross_selectors.add< eoRandomSelect<Bits> >();


    /***** Replacements ****/
    foundry.replacements.add< eoPlusReplacement<Bits> >();
    foundry.replacements.add< eoCommaReplacement<Bits> >();
    foundry.replacements.add< eoSSGAWorseReplacement<Bits> >();
    for(double i=0.51; i<0.92; i+=0.2) {
        foundry.replacements.add< eoSSGAStochTournamentReplacement<Bits> >(i);
    }
    for(size_t i=2; i < 11; i+=2) {
        foundry.replacements.add< eoSSGADetTournamentReplacement<Bits> >(i);
    }

    return foundry;
}

Bits::Fitness fake_func(const Bits&) { return 0; }

void print_irace_full(const eoParam& param, const size_t slot_size, std::string type="i", std::ostream& out = std::cout)
{
    assert(type == "i" or type =="c");

    // If there is no choice to be made on this operator, comment it out.
    if(slot_size - 1 <= 0) {
        out << "# ";
    }

    // irace doesn't support "-" in names.
    std::string irace_name = param.longName();
    irace_name.erase(std::remove(irace_name.begin(), irace_name.end(), '-'), irace_name.end());

    out << irace_name
        << "\t\"--" << param.longName() << "=\""
        << "\t" << type;

    if(type == "i") {
        if(slot_size -1 <= 0) {
            out << "\t(0)";
        } else {
            out << "\t(0," << slot_size-1 << ")";
        }
        out << std::endl;
    } else if(type == "c") {
        out << "\t(0";
        for(size_t i=1; i<slot_size; ++i) {
            out << "," << i;
        }
        out << ")" << std::endl;
    }
}

template<class ITF>
void print_irace_typed(const eoParam& param, const eoForgeVector<ITF>& op_foundry, std::ostream& out = std::cout)
{
    print_irace_full(param, op_foundry.size(), "c", out);
}

template<>
void print_irace_typed(const eoParam& param, const eoForgeVector<double>& op_foundry, std::ostream& out)
{
    print_irace_full(param, op_foundry.size(), "i", out);
}

template<class OPF>
void print_irace(const eoParam& param, const OPF& op_foundry, std::ostream& out = std::cout)
{
    print_irace_typed<typename OPF::Interface>(param, op_foundry, out);
}



void print_operator_typed(const eoFunctorBase& op, std::ostream& out)
{
    out << op.className();
}

void print_operator_typed(const double& op, std::ostream& out)
{
    out << op;
}

template<class OPF>
void print_operators(const eoParam& param, OPF& op_foundry, std::ostream& out = std::cout, std::string indent="  ")
{
    out << indent << op_foundry.size() << " " << param.longName() << ":" << std::endl;
    for(size_t i=0; i < op_foundry.size(); ++i) {
        out << indent << indent << i << ": ";
        auto& op = op_foundry.instantiate(i);
        print_operator_typed(op, out);
        out << std::endl;
    }
}


// Problem configuration.
struct Problem {
    double dummy;
    size_t epistasis;
    size_t neutrality;
    size_t ruggedness;
    size_t max_target;
    size_t dimension;
    friend std::ostream& operator<<(std::ostream& os, const Problem& pb);
};

std::ostream& operator<<(std::ostream& os, const Problem& pb)
{
    os << "u=" << pb.dummy << "_"
       << "e=" << pb.epistasis << "_"
       << "n=" << pb.neutrality << "_"
       << "r=" << pb.ruggedness << "_"
       << "t=" << pb.max_target << "_"
       << "d=" << pb.dimension;
    return os;
}

int main(int argc, char* argv[])
{
    /***** Global parameters. *****/
    enum { NO_ERROR = 0, ERROR_USAGE = 100 };

    std::map<size_t, Problem> problem_config_mapping {
       /* ┌ problem index in the map
        * │    ┌ problem ID in IOH experimenter
        * │    │     ┌ dummy
        * │    │     │   ┌ epistasis
        * │    │     │   │  ┌ neutrality
        * │    │     │   │  │    ┌ ruggedness
        * │    │     │   │  │    │   ┌ max target
        * │    │     │   │  │    │   │    ┌ dimension (bitstring length) */
        { 0 /* 1*/, {0,  6, 2,  10, 10,  20 }},
        { 1 /* 2*/, {0,  6, 2,  18, 10,  20 }},
        { 2 /* 3*/, {0,  5, 1,  72, 16,  16 }},
        { 3 /* 4*/, {0,  9, 3,  72, 16,  48 }},
        { 4 /* 5*/, {0, 23, 1,  90, 25,  25 }},
        { 5 /* 6*/, {0,  2, 1, 397, 32,  32 }},
        { 6 /* 7*/, {0, 11, 4,   0, 32, 128 }},
        { 7 /* 8*/, {0, 14, 4,   0, 32, 128 }},
        { 8 /* 9*/, {0,  8, 4, 128, 32, 128 }},
        { 9 /*10*/, {0, 36, 1, 245, 50,  50 }},
        {10 /*11*/, {0, 21, 2, 256, 50, 100 }},
        {11 /*12*/, {0, 16, 3, 613, 50, 150 }},
        {12 /*13*/, {0, 32, 2, 256, 64, 128 }},
        {13 /*14*/, {0, 21, 3,  16, 64, 192 }},
        {14 /*15*/, {0, 21, 3, 256, 64, 192 }},
        {15 /*16*/, {0, 21, 3, 403, 64, 192 }},
        {16 /*17*/, {0, 52, 4,   2, 64, 256 }},
        {17 /*18*/, {0, 60, 1,  16, 75,  75 }},
        {18 /*19*/, {0, 32, 2,   4, 75, 150 }}
    };

    eoFunctorStore store;

    eoParser parser(argc, argv, "FastGA interface for iRace");

    auto problem_p = parser.getORcreateParam<size_t>(0,
            "problem", "Problem ID",
            'p', "Problem", /*required=*/true);
    const size_t problem = problem_p.value();
    assert(0 <= problem and problem < problem_config_mapping.size());

    // const size_t dimension = parser.getORcreateParam<size_t>(1000,
    //         "dimension", "Dimension size",
    //         'd', "Problem").value();
    const size_t dimension = problem_config_mapping[problem].dimension;

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

    bool full_log =
        parser.getORcreateParam<bool>(0,
            "full-log", "Log the full search in CSV files (using the IOH profiler format)",
            'F').value();


    auto pop_size_p = parser.getORcreateParam<size_t>(1,
            "pop-size", "Population size",
            'P', "Operator Choice", /*required=*/false);
    const size_t pop_size = pop_size_p.value();

    auto instance_p = parser.getORcreateParam<size_t>(0,
            "instance", "Instance ID",
            'i', "Instance", /*required=*/false);
    const size_t instance = instance_p.value();

    auto continuator_p = parser.getORcreateParam<size_t>(0,
            "continuator", "Stopping criterion",
            'o', "Operator Choice", /*required=*/false); // Single alternative, not required.
    const size_t continuator = continuator_p.value();

    auto crossover_rate_p = parser.getORcreateParam<size_t>(0,
            "crossover-rate", "",
            'C', "Operator Choice", /*required=*/true);
    const size_t crossover_rate = crossover_rate_p.value();

    auto crossover_selector_p = parser.getORcreateParam<size_t>(0,
            "cross-selector", "How to selects candidates for cross-over",
            's', "Operator Choice", /*required=*/true);
    const size_t crossover_selector = crossover_selector_p.value();

    auto crossover_p = parser.getORcreateParam<size_t>(0,
            "crossover", "",
            'c', "Operator Choice", /*required=*/true);
    const size_t crossover = crossover_p.value();

    auto aftercross_selector_p = parser.getORcreateParam<size_t>(0,
            "aftercross-selector", "How to selects between the two individuals altered by cross-over which one will mutate",
            'a', "Operator Choice", /*required=*/false); // Single alternative, not required.
    const size_t aftercross_selector = aftercross_selector_p.value();

    auto mutation_rate_p = parser.getORcreateParam<size_t>(0,
            "mutation-rate", "",
            'M', "Operator Choice", /*required=*/true);
    const size_t mutation_rate = mutation_rate_p.value();

    auto mutation_selector_p = parser.getORcreateParam<size_t>(0,
            "mut-selector", "How to selects candidate for mutation",
            'u', "Operator Choice", /*required=*/true);
    const size_t mutation_selector = mutation_selector_p.value();

    auto mutation_p = parser.getORcreateParam<size_t>(0,
            "mutation", "",
            'm', "Operator Choice", /*required=*/true);
    const size_t mutation = mutation_p.value();

    auto replacement_p = parser.getORcreateParam<size_t>(0,
            "replacement", "",
            'r', "Operator Choice", /*required=*/true);
    const size_t replacement = replacement_p.value();

    auto offspring_size_p = parser.getORcreateParam<size_t>(0,
            "offspring-size", "Offsprings size (0 = same size than the parents pop, see --pop-size)",
            'O', "Operator Choice", /*required=*/false); // Single alternative, not required.
    const size_t offspring_size = offspring_size_p.value();


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

        std::clog << std::endl << "Available operators:" << std::endl;
        print_operators(        continuator_p, fake_foundry.continuators        , std::clog);
        print_operators(     crossover_rate_p, fake_foundry.crossover_rates     , std::clog);
        print_operators( crossover_selector_p, fake_foundry.crossover_selectors , std::clog);
        print_operators(aftercross_selector_p, fake_foundry.aftercross_selectors, std::clog);
        print_operators(          crossover_p, fake_foundry.crossovers          , std::clog);
        print_operators(      mutation_rate_p, fake_foundry.mutation_rates      , std::clog);
        print_operators(  mutation_selector_p, fake_foundry.mutation_selectors  , std::clog);
        print_operators(           mutation_p, fake_foundry.mutations           , std::clog);
        print_operators(        replacement_p, fake_foundry.replacements        , std::clog);
        print_operators(     offspring_size_p, fake_foundry.offspring_sizes     , std::clog);

        size_t n =
              fake_foundry.crossover_rates.size()
            * fake_foundry.crossover_selectors.size()
            * fake_foundry.crossovers.size()
            * fake_foundry.aftercross_selectors.size()
            * fake_foundry.mutation_rates.size()
            * fake_foundry.mutation_selectors.size()
            * fake_foundry.mutations.size()
            * fake_foundry.replacements.size()
            * fake_foundry.continuators.size()
            * fake_foundry.offspring_sizes.size();
        std::clog << std::endl;
        std::clog << n << " possible algorithms configurations." << std::endl;

        std::clog << "Ranges of configurable parameters (redirect the stdout in a file to use it with iRace): " << std::endl;

        // Do not print problem and instances, as they are managed separately by irace.
        std::cout << "# name\tswitch\ttype\trange" << std::endl;
        print_irace(        continuator_p, fake_foundry.continuators        , std::cout);
        print_irace(     crossover_rate_p, fake_foundry.crossover_rates     , std::cout);
        print_irace( crossover_selector_p, fake_foundry.crossover_selectors , std::cout);
        print_irace(aftercross_selector_p, fake_foundry.aftercross_selectors, std::cout);
        print_irace(          crossover_p, fake_foundry.crossovers          , std::cout);
        print_irace(      mutation_rate_p, fake_foundry.mutation_rates      , std::cout);
        print_irace(  mutation_selector_p, fake_foundry.mutation_selectors  , std::cout);
        print_irace(           mutation_p, fake_foundry.mutations           , std::cout);
        print_irace(        replacement_p, fake_foundry.replacements        , std::cout);
        print_irace(     offspring_size_p, fake_foundry.offspring_sizes     , std::cout);

        // std::ofstream irace_param("fastga.params");
        // irace_param << "# name\tswitch\ttype\tvalues" << std::endl;

        exit(NO_ERROR);
    }

    const size_t generations = static_cast<size_t>(std::floor(
                static_cast<double>(max_evals) / static_cast<double>(pop_size)));
    // const size_t generations = std::numeric_limits<size_t>::max();
    eo::log << eo::debug << "Number of generations: " << generations << std::endl;

    /***** IOH logger *****/


    auto max_target_para = problem_config_mapping[problem].max_target;
    IOHprofiler_RangeLinear<size_t> target_range(0, max_target_para, buckets);
    IOHprofiler_RangeLinear<size_t> budget_range(0, max_evals, buckets);
    IOHprofiler_ecdf_logger<int, size_t, size_t> ecdf_logger(
            target_range, budget_range,
            /*use_known_optimum*/false);

    // ecdf_logger.set_complete_flag(true);
    // ecdf_logger.set_interval(0);
    ecdf_logger.activate_logger();

    IOHprofiler_observer_combine<int> loggers(ecdf_logger);

    std::shared_ptr<IOHprofiler_csv_logger<int>> csv_logger;
    if(full_log) {
        // Build up an algorithm name from main parameters.
        std::ostringstream name;
        name << "FastGA";
        for(auto& p : {pop_size_p,
                crossover_rate_p,
                crossover_selector_p,
                crossover_p,
                aftercross_selector_p,
                mutation_rate_p,
                mutation_selector_p,
                mutation_p,
                replacement_p,
                offspring_size_p}) {
            name << "_" << p.shortName() << "=" << p.getValue();
        }
        std::clog << name.str() << std::endl;

        // Build up a problem description.
        std::ostringstream desc;
        desc << "pb=" << problem << "_";
        desc << problem_config_mapping[problem]; // Use the `operator<<` above.
        std::clog << desc.str() << std::endl;

        std::string dir(name.str());
        std::string folder(desc.str());
        csv_logger = std::make_shared<IOHprofiler_csv_logger<int>>(dir, folder, name.str(), desc.str());
        loggers.add(*csv_logger);
    }

    /***** IOH problem *****/
    double w_model_suite_dummy_para   = problem_config_mapping[problem].dummy;
    int w_model_suite_epitasis_para   = problem_config_mapping[problem].epistasis;
    int w_model_suite_neutrality_para = problem_config_mapping[problem].neutrality;
    int w_model_suite_ruggedness_para = problem_config_mapping[problem].ruggedness;

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
    w_model_om.IOHprofiler_set_problem_id(problem); // FIXME check what that means
    // w_model_om.IOHprofiler_set_instance_id(instance); // FIXME changing the instance seems to change the target upper bound.

    /// Set dimension.
    w_model_om.IOHprofiler_set_number_of_variables(dimension);

    /***** Bindings *****/
    ecdf_logger.track_problem(w_model_om);
    if(full_log) {
        csv_logger->track_problem(w_model_om);
    }

    eoEvalIOHproblem<Bits> onemax_pb(w_model_om, loggers);

    // eoEvalPrint<Bits> eval_print(onemax_pb, std::clog, "\n");
    eoEvalFuncCounter<Bits> eval_count(onemax_pb);

    eoPopLoopEval<Bits> onemax_eval(eval_count);

    /***** Instanciate and run the algo *****/

    eoBooleanGenerator<int> bgen;
    eoInitFixedLength<Bits> onemax_init(/*bitstring size=*/dimension, bgen);
    auto& foundry = make_foundry(store, onemax_init, eval_count, max_evals - pop_size, generations);

    Ints encoded_algo(foundry.size());

    encoded_algo[foundry.crossover_rates     .index()] = crossover_rate;
    encoded_algo[foundry.crossover_selectors .index()] = crossover_selector;
    encoded_algo[foundry.crossovers          .index()] = crossover;
    encoded_algo[foundry.aftercross_selectors.index()] = aftercross_selector;
    encoded_algo[foundry.mutation_rates      .index()] = mutation_rate;
    encoded_algo[foundry.mutation_selectors  .index()] = mutation_selector;
    encoded_algo[foundry.mutations           .index()] = mutation;
    encoded_algo[foundry.replacements        .index()] = replacement;
    encoded_algo[foundry.continuators        .index()] = continuator;
    encoded_algo[foundry.offspring_sizes     .index()] = offspring_size;

    // std::clog << "Encoded algorithm:" << std::endl;
    foundry.select(encoded_algo);
    std::clog << foundry.name() << std::endl;

    // // Evaluation of a forged encoded_algo on the sub-problem
    // eoEvalFoundryFastGA<Ints, Bits> eval_foundry(
    //         foundry, pop_size,
    //         onemax_init, onemax_eval,
    //         /*penalization=*/ dimension, // Worst case penalization.
    //         /*normalized=*/ false); // Use direct integer encoding.
    //
    // // Actually instanciate and run the algorithm.
    // eval_foundry(encoded_algo);

    eoPop<Bits> pop;
    pop.append(pop_size, onemax_init);
    onemax_eval(pop,pop);
    foundry(pop); // Actually run the selected algorithm.

    /***** IOH perf stats *****/
    IOHprofiler_ecdf_sum ecdf_sum;
    // iRace expects minimization
    long perf = ecdf_sum(ecdf_logger.data());

    // assert(0 < perf and perf <= buckets*buckets);
    if(perf <= 0 or buckets*buckets < perf) {
        std::cerr << "WARNING: illogical performance: " << perf
                  << ", check the bounds or the algorithm." << std::endl;
    }

    // std::clog << "After " << eval_count.getValue() << " / " << max_evals << " evaluations" << std::endl;

    // Output
    std::cout << -1 * perf << std::endl;

}
