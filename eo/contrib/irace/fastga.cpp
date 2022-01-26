#include <filesystem>
#include <iostream>
#include <cstdlib>
#include <string>
#include <memory>

#include <eo>
#include <ga.h>
#include <utils/checkpointing>
#include <eoInt.h>
#include <problems/eval/eoEvalIOH.h>

#include <ioh.hpp>

/*****************************************************************************
 * ParadisEO algorithmic grammar definition.
 *****************************************************************************/

using Bits = eoBit<eoMaximizingFitnessT<int>, int>;

// by enumerating candidate operators and their parameters.
eoAlgoFoundryFastGA<Bits>& make_foundry(
        eoFunctorStore& store,
        eoInit<Bits>& init,
        eoEvalFunc<Bits>& eval,
        const size_t max_evals,
        const size_t generations,
        const double optimum
    )
{
    // FIXME using max_restarts>1 does not allow to honor max evals.
    auto& foundry = store.pack< eoAlgoFoundryFastGA<Bits> >(init, eval, max_evals, /*max_restarts=*/1);

    /***** Continuators ****/
    auto& fitcont = store.pack< eoFitContinue<Bits> >(optimum);
    auto& gencont = store.pack< eoGenContinue<Bits> >(generations);
    auto combconts = std::make_shared< std::vector<eoContinue<Bits>*> >();
    combconts->push_back( &fitcont );
    combconts->push_back( &gencont );
    foundry.continuators.add< eoCombinedContinue<Bits> >( *combconts );
    // for(size_t i=1; i<10; i++) {
    //     foundry.continuators.add< eoGenContinue<Bits> >(i);
    // }
    // for(size_t i=10; i < 100; i+=2 ) {
    //     foundry.continuators.add< eoSteadyFitContinue<Bits> >(10,i);
    // }

    // for(double i=0.0; i<1.0; i+=0.2) {
    //     foundry.crossover_rates.add<double>(i);
    //     foundry.mutation_rates.add<double>(i);
    // }

    /***** Offsprings size *****/
    // for(size_t i=5; i<100; i+=10) {
    //     foundry.offspring_sizes.add<size_t>(i);
    // }

    foundry.offspring_sizes.setup(0,100); // 0 = use parents fixed pop size.

    /***** Crossovers ****/
    for(double i=0.1; i<1.0; i+=0.2) {
        foundry.crossovers.add< eoUBitXover<Bits> >(i); // preference over 1
    }
    for(size_t i=1; i < 10; i+=2) {

        foundry.crossovers.add< eoNPtsBitXover<Bits> >(i); // nb of points
    }
    // foundry.crossovers.add< eo1PtBitXover<Bits> >(); // Same as NPts=1

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

/*****************************************************************************
 * irace helper functions.
 *****************************************************************************/

Bits::Fitness fake_func(const Bits&) { return 0; }

void print_irace_categorical(const eoParam& param, const size_t slot_size, std::string type="c", std::ostream& out = std::cout)
{
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

    out << "\t(0";
    for(size_t i=1; i<slot_size; ++i) {
        out << "," << i;
    }
    out << ")" << std::endl;
}

template<class T>
void print_irace_ranged(const eoParam& param, const T min, const T max, std::string type="r", std::ostream& out = std::cout)
{
    // If there is no choice to be made on this operator, comment it out.
    if(max - min <= 0) {
        out << "# ";
    }

    // irace doesn't support "-" in names.
    std::string irace_name = param.longName();
    irace_name.erase(std::remove(irace_name.begin(), irace_name.end(), '-'), irace_name.end());

    out << irace_name
        << "\t\"--" << param.longName() << "=\""
        << "\t" << type;

    if(max-min <= 0) {
        out << "\t(?)";
    } else {
        out << "\t(" << min << "," << max << ")";
    }
    out << std::endl;
}


template<class ITF>
void print_irace_oper(const eoParam& param, const eoOperatorFoundry<ITF>& op_foundry, std::ostream& out = std::cout)
{
    print_irace_categorical(param, op_foundry.size(), "c", out);
}

// FIXME generalize to any scalar type with enable_if
// template<class ITF>
void print_irace_param(
    const eoParam& param,
    // const eoParameterFoundry<typename std::enable_if< std::is_floating_point<ITF>::value >::type>& op_foundry,
    const eoParameterFoundry<double>& op_foundry,
    std::ostream& out)
{
    print_irace_ranged(param, op_foundry.min(), op_foundry.max(), "r", out);
}

// template<class ITF>
void print_irace_param(
    const eoParam& param,
    // const eoParameterFoundry<typename std::enable_if< std::is_integral<ITF>::value >::type>& op_foundry,
    const eoParameterFoundry<size_t>& op_foundry,
    std::ostream& out)
{
    print_irace_ranged(param, op_foundry.min(), op_foundry.max(), "i", out);
}


template<class ITF>
void print_irace(const eoParam& param, const eoOperatorFoundry<ITF>& op_foundry, std::ostream& out = std::cout)
{
    print_irace_oper<ITF>(param, op_foundry, out);
}

template<class ITF>
void print_irace(const eoParam& param, const eoParameterFoundry<ITF>& op_foundry, std::ostream& out = std::cout)
{
    print_irace_param/*<ITF>*/(param, op_foundry, out);
}

void print_irace(const eoParam& param, const size_t min, const size_t max, std::ostream& out = std::cout)
{
    print_irace_ranged(param, min, max, "i", out);
}

void print_operator_typed(const eoFunctorBase& op, std::ostream& out)
{
    out << op.className();
}

void print_operator_typed(const double& op, std::ostream& out)
{
    out << op;
}

template<class ITF>
void print_operators(const eoParam& param, eoOperatorFoundry<ITF>& op_foundry, std::ostream& out = std::cout, std::string indent="  ")
{
    out << indent << op_foundry.size() << " " << param.longName() << ":" << std::endl;
    for(size_t i=0; i < op_foundry.size(); ++i) {
        out << indent << indent << i << ": ";
        auto& op = op_foundry.instantiate(i);
        print_operator_typed(op, out);
        out << std::endl;
    }
}

template<class T>
void print_operators(const eoParam& param, T min, T max, std::ostream& out = std::cout, std::string indent="  ")
{
    out << indent << "[" << min << "," << max << "] " << param.longName() << "." << std::endl;
}

template<class ITF>
void print_operators(const eoParam& param, eoParameterFoundry<ITF>& op_foundry, std::ostream& out = std::cout, std::string indent="  ")
{
    print_operators(param, op_foundry.min(), op_foundry.max(), out, indent);
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

/*****************************************************************************
 * IOH problem adaptation.
 *****************************************************************************/

class WModelFlat : public ioh::problem::wmodel::WModelOneMax
{
    public:
        WModelFlat(const int instance, const int n_variables,
                   const double dummy_para, const int epistasis_para, const int neutrality_para,
                   const int ruggedness_para)
        : WModelOneMax(instance, n_variables, dummy_para, epistasis_para, neutrality_para, ruggedness_para)
        { }

    protected:
        double transform_objectives(const double y) override
        { // Disable objective function shift & scaling.
            return y;
        }
};

/*****************************************************************************
 * Command line interface.
 *****************************************************************************/

int main(int argc, char* argv[])
{
    /***** Global parameters. *****/
    enum { NO_ERROR = 0, ERROR_USAGE = 100 };

    std::map<size_t, Problem> benchmark {
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

    /***** Problem parameters *****/
    auto problem_p = parser.getORcreateParam<size_t>(0,
            "problem", "Problem ID",
            'p', "Problem", /*required=*/true);
    const size_t problem = problem_p.value();
    assert(problem < benchmark.size());

    // const size_t dimension = parser.getORcreateParam<size_t>(1000,
    //         "dimension", "Dimension size",
    //         'd', "Problem").value();
    const size_t dimension = benchmark[problem].dimension;

    auto instance_p = parser.getORcreateParam<size_t>(0,
            "instance", "Instance ID",
            'i', "Instance", /*required=*/false);
    const size_t instance = instance_p.value();

    const size_t max_evals = parser.getORcreateParam<size_t>(5 * dimension,
            "max-evals", "Maximum number of evaluations (default: 5*dim, else the given value)",
            'e', "Stopping criterion").value();

    const size_t buckets = parser.getORcreateParam<size_t>(100,
            "buckets", "Number of buckets for discretizing the ECDF",
            'b', "Performance estimation").value();

    /***** Generic options *****/
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
            "full-log", "Log the full search in CSV files"/* (using the IOH profiler format)"*/,
            'F').value();

    bool output_mat =
        parser.getORcreateParam<bool>(0,
            "output-mat", "Output the aggregated attainment matrix instead of its scalar sum (fancy colormap on stderr, parsable CSV on stdout).",
            'A').value();

    /***** populations sizes *****/
    auto pop_size_p = parser.getORcreateParam<size_t>(5,
            "pop-size", "Population size",
            'P', "Operator Choice", /*required=*/false);
    const size_t pop_size = pop_size_p.value();
    const size_t pop_size_max = 200;

    auto offspring_size_p = parser.getORcreateParam<size_t>(0,
            "offspring-size", "Offsprings size (0 = same size than the parents pop, see --pop-size)",
            'O', "Operator Choice", /*required=*/false); // Single alternative, not required.
    const size_t offspring_size = offspring_size_p.value();

    size_t generations = static_cast<size_t>(std::floor(
                static_cast<double>(max_evals) / static_cast<double>(pop_size)));
    // const size_t generations = std::numeric_limits<size_t>::max();
    if(generations < 1) {
        generations = 1;
    }

    /***** metric parameters *****/
    auto crossover_rate_p = parser.getORcreateParam<double>(0.5,
            "crossover-rate", "",
            'C', "Operator Choice", /*required=*/true);
    const double crossover_rate = crossover_rate_p.value();

    auto mutation_rate_p = parser.getORcreateParam<double>(0,
            "mutation-rate", "",
            'M', "Operator Choice", /*required=*/true);
    const double mutation_rate = mutation_rate_p.value();

    /***** operators *****/
    auto continuator_p = parser.getORcreateParam<size_t>(0,
            "continuator", "Stopping criterion",
            'o', "Operator Choice", /*required=*/false); // Single alternative, not required.
    const size_t continuator = continuator_p.value();

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

    // Help + Verbose routines
    make_verbose(parser);
    make_help(parser, /*exit_after*/false, std::clog);

    if(parser.userNeedsHelp()) {

        // Fake operators, just to be able to call make_foundry
        // to get the configured operators slots.
        eoEvalFuncPtr<Bits> fake_eval(fake_func);
        eoUniformGenerator<int> fake_gen(0, 1);
        eoInitFixedLength<Bits> fake_init(/*bitstring size=*/1, fake_gen);
        auto fake_foundry = make_foundry(store, fake_init, fake_eval, max_evals, /*generations=*/ 1, 0);

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
        print_operators(           pop_size_p, (size_t)1, pop_size_max          , std::clog);
        std::clog << std::endl;

        // If we were to make a DoE sampling numeric parameters,
        // we would use that many samples:
        size_t fake_sample_size = 10;
        std::clog << "With " << fake_sample_size << " samples for numeric parameters..." << std::endl;
        size_t n =
              fake_sample_size //crossover_rates
            * fake_foundry.crossover_selectors.size()
            * fake_foundry.crossovers.size()
            * fake_foundry.aftercross_selectors.size()
            * fake_sample_size //mutation_rates
            * fake_foundry.mutation_selectors.size()
            * fake_foundry.mutations.size()
            * fake_foundry.replacements.size()
            * fake_foundry.continuators.size()
            * fake_sample_size //offspring_sizes
            * fake_sample_size //pop_size
            ;
        std::clog << "~" << n << " possible algorithms configurations." << std::endl;

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
        print_irace(           pop_size_p, 1, pop_size_max                  , std::cout);

        // std::ofstream irace_param("fastga.params");
        // irace_param << "# name\tswitch\ttype\tvalues" << std::endl;

        exit(NO_ERROR);
    }

    eo::log << eo::debug << "Maximum number of evaluations: " << max_evals << std::endl;
    eo::log << eo::debug << "Number of generations: " << generations << std::endl;


    /*****************************************************************************
     * IOH stuff.
     *****************************************************************************/

    /***** IOH logger *****/
    auto max_target = benchmark[problem].max_target;
    ioh::logger::eah::Log10Scale<double> target_range(0, max_target, buckets);
    ioh::logger::eah::Log10Scale<size_t> budget_range(0, max_evals, buckets);
    ioh::logger::EAH eah_logger(target_range, budget_range);

    ioh::logger::Combine loggers(eah_logger);

    std::shared_ptr<ioh::logger::FlatFile> csv_logger = nullptr;
    if(full_log) {
        // Build up an algorithm name from main parameters.
        std::ostringstream name;
        name << "FastGA";
        for(auto& p : {
                crossover_selector_p,
                crossover_p,
                aftercross_selector_p,
                mutation_selector_p,
                mutation_p,
                replacement_p }) {
            name << "_" << p.shortName() << "=" << p.getValue();
        }
        for(auto& p : {
                crossover_rate_p,
                mutation_rate_p }) {
            name << "_" << p.shortName() << "=" << p.getValue();
        }
        for(auto& p : {pop_size_p,
                offspring_size_p }) {
            name << "_" << p.shortName() << "=" << p.getValue();
        }
        std::clog << name.str() << std::endl;

        // Build up a problem description.
        std::ostringstream desc;
        desc << "pb=" << problem << "_";
        desc << benchmark[problem]; // Use the `operator<<` above.
        std::clog << desc.str() << std::endl;

        std::filesystem::path folder = desc.str();
        std::filesystem::create_directories(folder);

        ioh::trigger::OnImprovement on_improvement;
        ioh::watch::Evaluations evaluations;
        ioh::watch::TransformedYBest transformed_y_best;
        std::vector<std::reference_wrapper<ioh::logger::Trigger >> t = {on_improvement};
        std::vector<std::reference_wrapper<ioh::logger::Property>> w = {evaluations,transformed_y_best};
        csv_logger = std::make_shared<ioh::logger::FlatFile>(
            // {std::ref(on_improvement)},
            // {std::ref(evaluations),std::ref(transformed_y_best)},
            t, w,
            name.str(),
            folder
        );
        loggers.append(*csv_logger);
    }

    /***** IOH problem *****/
    double w_dummy   = benchmark[problem].dummy;
    int w_epitasis   = benchmark[problem].epistasis;
    int w_neutrality = benchmark[problem].neutrality;
    int w_ruggedness = benchmark[problem].ruggedness;

    // std::string problem_name = "OneMax";
    // problem_name = problem_name
    //                 + "_D" + std::to_string((int)(w_dummy * dimension))
    //                 + "_E" + std::to_string(w_epitasis)
    //                 + "_N" + std::to_string(w_neutrality)
    //                 + "_R" + std::to_string(w_ruggedness);

    // ioh::problem::wmodel::WModelOneMax w_model_om(
    WModelFlat w_model_om(
        instance,
        dimension, 
        w_dummy,
        w_epitasis,
        w_neutrality,
        w_ruggedness);

    /***** Bindings *****/
    w_model_om.attach_logger(loggers);

    /*****************************************************************************
     * Binding everything together.
     *****************************************************************************/

    eoEvalIOHproblem<Bits> onemax_pb(w_model_om, loggers);

    // eoEvalPrint<Bits> eval_print(onemax_pb, std::clog, "\n");
    // eoEvalFuncCounter<Bits> eval_count(onemax_pb);
    eoEvalCounterThrowException<Bits> eval_count(onemax_pb, max_evals);

    eoPopLoopEval<Bits> onemax_eval(eval_count);

    /***** Instanciate and run the algo *****/

    eoBooleanGenerator<int> bgen;
    eoInitFixedLength<Bits> onemax_init(/*bitstring size=*/dimension, bgen);
    auto& foundry = make_foundry(store, onemax_init, eval_count, max_evals, generations, max_target);

    eoAlgoFoundry<Bits>::Encodings encoded_algo(foundry.size());

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

    /*****************************************************************************
     * Run and output results.
     *****************************************************************************/

    eoPop<Bits> pop;
    pop.append(pop_size, onemax_init);
    try {
        onemax_eval(pop,pop);
        foundry(pop); // Actually run the selected algorithm.
        
    } catch(eoMaxEvalException & e) {
        eo::log << eo::debug << "Reached maximum evaluations: " << eval_count.getValue() << " / " << max_evals << std::endl;
    }

    /***** IOH perf stats *****/
    double perf = ioh::logger::eah::stat::under_curve::volume(eah_logger);

    if(perf == 0 or perf > max_target * max_evals * 1.0) {
        std::cerr << "WARNING: illogical performance? " << perf
                  << " Check the bounds or the algorithm." << std::endl;
    }

    // std::clog << "After " << eval_count.getValue() << " / " << max_evals << " evaluations" << std::endl;

    if(output_mat) {
        std::vector<std::vector<double>> mat = ioh::logger::eah::stat::distribution(eah_logger);

        // Fancy color map on clog.
        std::clog << ioh::logger::eah::colormap(mat) << std::endl;

        // Parsable CSV on cout.
        std::clog << "Attainment matrix distribution: " << std::endl;
        assert(mat.size() > 0);
        assert(mat[0].size() > 1);
        for(size_t i = mat.size()-1; i > 0; --i) {
            assert(mat[i].size() >= 1);
            std::cout << mat[i][0];
            for(size_t j = 1; j < mat[i].size(); ++j) {
                std::cout << "," << mat[i][j];
            }
            std::cout << std::endl;
        }

    } else {
        // iRace expects minimization
        std::cout << -1 * perf << std::endl;
    }
}
