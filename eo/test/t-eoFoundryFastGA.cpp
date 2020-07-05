#include <iostream>
#include <string>

#include <eo>
#include <ga.h>
#include <utils/checkpointing>
#include "../../problems/eval/oneMaxEval.h"

using Particle = eoRealParticle<eoMaximizingFitness>;
using Bits = eoBit<double>;

// Generate a search space of 5,232,000 algorithms,
// by enumerating candidate operators and their parameters.
eoAlgoFoundryFastGA<Bits>& make_foundry(eoFunctorStore& store, eoInit<Bits>& init, eoEvalFunc<Bits>& eval_onemax)
{
    auto& foundry = store.pack< eoAlgoFoundryFastGA<Bits> >(init, eval_onemax, 20,10);

    /***** Continuators ****/
    for(size_t i=10; i < 100; i+=2 ) {
        foundry.continuators.add< eoSteadyFitContinue<Bits> >(10,i);
    }

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


int main(int /*argc*/, char** /*argv*/)
{
    eo::log << eo::setlevel(eo::warnings);
    eoFunctorStore store;

    oneMaxEval<Bits> onemax_eval;

    eoBooleanGenerator gen(0.5);
    eoInitFixedLength<Bits> init(/*bitstring size=*/5, gen);

    auto& foundry = make_foundry(store, init, onemax_eval);

    size_t n = foundry.continuators.size() * foundry.crossovers.size() * foundry.mutations.size() * foundry.selectors.size() * foundry.replacements.size()* foundry.crossover_rates.size() * foundry.mutation_rates.size() * foundry.pop_sizes.size();
    std::clog << n << " possible algorithms instances." << std::endl;

    eoPop<Bits> pop;
    pop.append(5,init);
    ::apply(onemax_eval,pop);

    foundry.select({0,0,0,0,0,0,0,0});
    foundry(pop);

    std::cout << "Done" << std::endl;
    std::cout << pop << std::endl;
    std::cout << pop.best_element() << std::endl;
}
