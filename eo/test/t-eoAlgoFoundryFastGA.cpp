#include <iostream>
#include <string>

#include <eo>
#include <ga.h>
#include "../../problems/eval/oneMaxEval.h"


int main(int /*argc*/, char** /*argv*/)
{
    size_t dim = 500;
    size_t pop_size = 10;

    eo::log << eo::setlevel(eo::warnings);

    using EOT = eoBit<double>;

    oneMaxEval<EOT> eval;

    eoBooleanGenerator gen(0.5);
    eoInitFixedLength<EOT> init(dim, gen);

    eoAlgoFoundryFastGA<EOT> foundry(init, eval, pop_size*10);

    /***** Variation rates *****/
    for(double r = 0.0; r < 1.0; r+=0.1) {
        foundry.crossover_rates.add<double>(r);
        foundry. mutation_rates.add<double>(r);
    }

    /***** Crossovers ****/
    foundry.crossovers.add< eo1PtBitXover<EOT> >();
    foundry.crossovers.add< eoUBitXover<EOT> >(0.5); // preference over 1
    for(size_t i=1; i < 11; i+=4) {
        foundry.crossovers.add< eoNPtsBitXover<EOT> >(i); // nb of points
    }

    /***** Mutations ****/
    foundry.mutations.add< eoBitMutation<EOT> >(0.01); // proba of flipping one bit
    for(size_t i=1; i < 11; i+=4) {
        foundry.mutations.add< eoDetBitFlip<EOT> >(i); // mutate k bits
    }

    /***** Selectors *****/
    for(eoOperatorFoundry<eoSelectOne<EOT>>& ops :
        {std::ref(foundry.crossover_selectors),
         std::ref(foundry.aftercross_selectors),
         std::ref(foundry.mutation_selectors) }) {

        ops.add< eoRandomSelect<EOT> >();
        ops.add< eoStochTournamentSelect<EOT> >(0.5);
        ops.add< eoSequentialSelect<EOT> >();
        ops.add< eoProportionalSelect<EOT> >();
        for(size_t i=2; i < 10; i+=4) {
            ops.add< eoDetTournamentSelect<EOT> >(i);
        }
    }

    /***** Replacements ****/
    foundry.replacements.add< eoCommaReplacement<EOT> >();
    foundry.replacements.add< eoPlusReplacement<EOT> >();
    foundry.replacements.add< eoSSGAWorseReplacement<EOT> >();
    foundry.replacements.add< eoSSGAStochTournamentReplacement<EOT> >(0.51);
    for(size_t i=2; i < 10; i+=4) {
        foundry.replacements.add< eoSSGADetTournamentReplacement<EOT> >(i);
    }

    /***** Continuators ****/
    for(size_t i=10; i < 30; i+=10 ) {
        foundry.continuators.add< eoSteadyFitContinue<EOT> >(10,i);
    }

    /***** Offspring population size *****/
    foundry.offspring_sizes.add<size_t>(0); // 0 = same as parent pop
    // for(size_t s = pop_size; s < 2*pop_size; s+=pop_size/10) {
    //     foundry.offspring_sizes.add<size_t>(s);
    // }


    size_t n =
          foundry.crossover_rates.size()
        * foundry.crossover_selectors.size()
        * foundry.crossovers.size()
        * foundry.aftercross_selectors.size()
        * foundry.mutation_rates.size()
        * foundry.mutation_selectors.size()
        * foundry.mutations.size()
        * foundry.replacements.size()
        * foundry.continuators.size()
        * foundry.offspring_sizes.size();
    std::clog << n << " possible algorithms instances." << std::endl;

    EOT best_sol;
    std::string best_algo = "";

    size_t i=0;
    for(size_t i_crossrate = 0; i_crossrate < foundry.crossover_rates.size(); ++i_crossrate ) {
        for(size_t i_crossselect = 0; i_crossselect < foundry.crossover_selectors.size(); ++i_crossselect ) {
            for(size_t i_cross = 0; i_cross < foundry.crossovers.size(); ++i_cross ) {
                for(size_t i_aftercrosel = 0; i_aftercrosel < foundry.aftercross_selectors.size(); ++i_aftercrosel ) {
                    for(size_t i_mutrate = 0; i_mutrate < foundry.mutation_rates.size(); ++i_mutrate ) {
                        for(size_t i_mutselect = 0; i_mutselect < foundry.mutation_selectors.size(); ++i_mutselect ) {
                            for(size_t i_mut = 0; i_mut < foundry.mutations.size(); ++i_mut ) {
                                for(size_t i_rep = 0; i_rep < foundry.replacements.size(); ++i_rep ) {
                                    for(size_t i_cont = 0; i_cont < foundry.continuators.size(); ++i_cont ) {
                                        for(size_t i_pop = 0; i_pop < foundry.offspring_sizes.size(); ++i_pop ) {
                                            std::clog << "\r" << i++ << "/" << n-1; std::clog.flush();

                                            eoPop<EOT> pop;
                                            pop.append(pop_size, init);

                                            foundry.select({
                                                i_crossrate,
                                                i_crossselect,
                                                i_cross,
                                                i_aftercrosel,
                                                i_mutrate,
                                                i_mutselect,
                                                i_mut,
                                                i_rep,
                                                i_cont,
                                                i_pop
                                            });

                                            // Actually perform a search
                                            foundry(pop);

                                            if(best_sol.invalid()) {
                                                best_sol = pop.best_element();
                                                best_algo = foundry.name();
                                            } else if(pop.best_element().fitness() > best_sol.fitness()) {
                                                best_sol = pop.best_element();
                                                best_algo = foundry.name();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << std::endl << "Best algo: " << best_algo << ", with " << best_sol << std::endl;

}
