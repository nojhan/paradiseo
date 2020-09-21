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


int main(int argc, char* argv[])
{
    /***** Global parameters. *****/
    enum { ERROR_USAGE = 100 };

    const size_t dimension = 1000;
    const size_t max_evals = 2 * dimension;
    const size_t buckets = 100;

    eoFunctorStore store;

    /***** Parameters managed by the caller. *****/
    if(argc != 11) {
        // Fake operators, just to be able to call make_foundry
        // to get the configured operators slots.
        eoEvalFuncPtr<Bits> fake_eval(fake_func);
        eoUniformGenerator<int> fake_gen(0, 1);
        eoInitFixedLength<Bits> fake_init(/*bitstring size=*/1, fake_gen);
        auto fake_foundry = make_foundry(store, fake_init, fake_eval, max_evals, /*generations=*/ 1);

        std::cerr << "Usage: " << argv[0] << std::endl;
        std::cerr << "\t<pb_instance>    in [0,18]" << std::endl;
        std::cerr << "\t<random_seed>    in [0,MAXULONG[" << std::endl;
        std::cerr << "\t<continuator>    in [0," << fake_foundry.continuators   .size() << "[" << std::endl;
        std::cerr << "\t<crossover_rate> in [0," << fake_foundry.crossover_rates.size() << "[" << std::endl;
        std::cerr << "\t<crossover>      in [0," << fake_foundry.crossovers     .size() << "[" << std::endl;
        std::cerr << "\t<mutation_rate>  in [0," << fake_foundry.mutation_rates .size() << "[" << std::endl;
        std::cerr << "\t<mutation>       in [0," << fake_foundry.mutations      .size() << "[" << std::endl;
        std::cerr << "\t<selector>       in [0," << fake_foundry.selectors      .size() << "[" << std::endl;
        std::cerr << "\t<pop_size>       in [0," << fake_foundry.pop_sizes      .size() << "[" << std::endl;
        std::cerr << "\t<replacement>    in [0," << fake_foundry.replacements   .size() << "[" << std::endl;

        exit(ERROR_USAGE);
    }

    const int pb_instance = std::atoi(argv[1]);

    std::string s(argv[2]);
    eo::rng.reseed(std::stoull(s));

    Ints encoded_algo(8);
    encoded_algo[0] = std::atoi(argv[3]); // continuator
    encoded_algo[1] = std::atoi(argv[4]); // crossover_rate
    encoded_algo[2] = std::atoi(argv[5]); // crossover
    encoded_algo[3] = std::atoi(argv[6]); // mutation_rate
    encoded_algo[4] = std::atoi(argv[7]); // mutation
    encoded_algo[5] = std::atoi(argv[8]); // selection
    encoded_algo[6] = std::atoi(argv[9]); // pop_size
    encoded_algo[7] = std::atoi(argv[10]); // replacement

    const size_t pop_size = encoded_algo[6];
    const size_t generations = max_evals / pop_size;

    eo::log << eo::setlevel(eo::warnings);

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
    w_model_om.IOHprofiler_set_instance_id(pb_instance);

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

    size_t n = foundry.continuators.size() * foundry.crossovers.size() * foundry.mutations.size() * foundry.selectors.size() * foundry.replacements.size();
    std::clog << n << " possible algorithms instances." << std::endl;

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
