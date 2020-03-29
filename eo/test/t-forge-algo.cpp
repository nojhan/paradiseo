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

    eoGenContinue<EOT> common_cont(100);

    eoForgeVector< eoContinue<EOT> > continuators;
    continuators.add< eoSteadyFitContinue<EOT> >(10,10);
    continuators.add< eoGenContinue<EOT> >(100);

    eoForgeVector< eoQuadOp<EOT> > crossovers;
    crossovers.add< eo1PtBitXover<EOT> >();
    crossovers.add< eoUBitXover<EOT> >(0.5); // preference over 1
    crossovers.add< eoNPtsBitXover<EOT> >(2); // nb of points

    eoForgeVector< eoMonOp<EOT> > mutations;
    mutations.add< eoBitMutation<EOT> >(0.01); // proba of flipping one bit
    mutations.add< eoDetBitFlip<EOT> >(1); // mutate k bits

    eoForgeVector< eoSelectOne<EOT> > selectors;
    selectors.add< eoDetTournamentSelect<EOT> >(pop_size/2);
    selectors.add< eoStochTournamentSelect<EOT> >(0.5);
    selectors.add< eoSequentialSelect<EOT> >();
    selectors.add< eoProportionalSelect<EOT> >();

    eoForgeVector< eoReplacement<EOT> > replacors;
    replacors.add< eoCommaReplacement<EOT> >();
    replacors.add< eoPlusReplacement<EOT> >();
    replacors.add< eoSSGAWorseReplacement<EOT> >();
    replacors.add< eoSSGADetTournamentReplacement<EOT> >(pop_size/2);
    replacors.add< eoSSGAStochTournamentReplacement<EOT> >(0.51);

    std::clog << continuators.size() * crossovers.size() * mutations.size() * selectors.size() * replacors.size()
              << " possible algorithms instances." << std::endl;

    EOT best_sol;
    std::string best_algo = "";

    for(auto& forge_cont : continuators) {
        auto& continuator = forge_cont->instanciate();

        for(auto& forge_cross : crossovers) {
            auto& crossover = forge_cross->instanciate();

            for(auto& forge_mut : mutations ) {
                auto& mutation = forge_mut->instanciate();

                for(auto& forge_sel : selectors) {
                    auto& selector = forge_sel->instanciate();

                    for(auto& forge_rep : replacors) {
                        auto& replacor = forge_rep->instanciate();

                        std::ostringstream algo_name;
                        algo_name << continuator.className() << " + "
                                  << crossover.className()   << " + "
                                  << mutation.className()    << " + "
                                  << selector.className()    << " + "
                                  << replacor.className();

                        std::clog << "ALGO: " << algo_name.str();
                        std::clog.flush();

                        eoSequentialOp<EOT> variator;
                        variator.add(crossover, 1.0);
                        variator.add(mutation, 1.0);

                        eoGeneralBreeder<EOT> breeder(selector, variator, 1.0);

                        eoCombinedContinue<EOT> gen_cont(common_cont);
                        gen_cont.add(continuator);

                        eoEasyEA<EOT> algo(gen_cont, eval, breeder, replacor);

                        eoPop<EOT> pop;
                        pop.append(pop_size, init);
                        apply(eval,pop);

                        algo(pop);

                        std::clog << " = " << pop.best_element().fitness() << std::endl;

                        if(best_sol.invalid()) {
                           best_sol = pop.best_element();
                           best_algo = algo_name.str();
                        } else if(pop.best_element().fitness() > best_sol.fitness()) {
                            best_sol = pop.best_element();
                            best_algo = algo_name.str();
                        }

                    }
                }
            }
        }
    }
    std::cout << "Best algo: " << best_algo << ", with " << best_sol << std::endl;

}
