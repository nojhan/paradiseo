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

    oneMaxEval<EOT> evalfunc;
    eoPopLoopEval<EOT> eval(evalfunc);

    eoBooleanGenerator gen(0.5);
    eoInitFixedLength<EOT> init(dim, gen);

    eoAlgoFoundryEA<EOT> foundry(eval, 10);

    /***** Continuators ****/
    for(size_t i=10; i < 30; i+=10 ) {
        foundry.continuators.add< eoSteadyFitContinue<EOT> >(10,i);
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
    foundry.selectors.add< eoStochTournamentSelect<EOT> >(0.5);
    foundry.selectors.add< eoSequentialSelect<EOT> >();
    foundry.selectors.add< eoProportionalSelect<EOT> >();
    for(size_t i=2; i < 10; i+=4) {
        foundry.selectors.add< eoDetTournamentSelect<EOT> >(i);
    }

    /***** Replacements ****/
    foundry.replacements.add< eoCommaReplacement<EOT> >();
    foundry.replacements.add< eoPlusReplacement<EOT> >();
    foundry.replacements.add< eoSSGAWorseReplacement<EOT> >();
    foundry.replacements.add< eoSSGAStochTournamentReplacement<EOT> >(0.51);
    for(size_t i=2; i < 10; i+=4) {
        foundry.replacements.add< eoSSGADetTournamentReplacement<EOT> >(i);
    }


    size_t n = foundry.continuators.size() * foundry.crossovers.size() * foundry.mutations.size() * foundry.selectors.size() * foundry.replacements.size();
    std::clog << n << " possible algorithms instances." << std::endl;

    EOT best_sol;
    std::string best_algo = "";

    size_t i=0;
    for(size_t i_cont = 0; i_cont < foundry.continuators.size(); ++i_cont ) {
        for(size_t i_cross = 0; i_cross < foundry.crossovers.size(); ++i_cross ) {
            for(size_t i_mut = 0; i_mut < foundry.mutations.size(); ++i_mut ) {
                for(size_t i_sel = 0; i_sel < foundry.selectors.size(); ++i_sel ) {
                    for(size_t i_rep = 0; i_rep < foundry.replacements.size(); ++i_rep ) {
                        std::clog << "\r" << i++ << "/" << n-1; std::clog.flush();

                        eoPop<EOT> pop;
                        pop.append(pop_size, init);
                        eval(pop,pop);

                        foundry.select({
                                size_t{i_cont},
                                size_t{i_cross},
                                size_t{i_mut},
                                size_t{i_sel},
                                size_t{i_rep}
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
    std::cout << std::endl << "Best algo: " << best_algo << ", with " << best_sol << std::endl;

}
