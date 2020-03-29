#include <iostream>
#include <string>

#include <eo>
#include <ga.h>
#include "../../problems/eval/oneMaxEval.h"

template<class EOT>
class eoFoundryEA : public eoAlgo<EOT>
{
    public:
        static const size_t dim = 5;

    protected:
        std::array<size_t, dim> _encoding;

        struct Indices
        {
            static const size_t continuators = 0;
            static const size_t crossovers = 1;
            static const size_t mutations = 2;
            static const size_t selectors = 3;
            static const size_t replacements = 4;
        };

    public:
        const Indices index_of;

        eoFoundryEA( eoEvalFunc<EOT>& eval ) :
            index_of(),
            _eval(eval)
        {
            _encoding = { 0 }; // dim * 0
        }

        size_t& at(size_t i)
        {
            return _encoding.at(i);
        }

        void operator=( std::array<size_t,dim> a)
        {
            _encoding = a;
        }

        eoForgeVector< eoContinue<EOT>    > continuators;
        eoForgeVector< eoQuadOp<EOT>      > crossovers;
        eoForgeVector< eoMonOp<EOT>       > mutations;
        eoForgeVector< eoSelectOne<EOT>   > selectors;
        eoForgeVector< eoReplacement<EOT> > replacements;

        void operator()(eoPop<EOT>& pop)
        {
            assert(continuators.size() > 0); assert(_encoding.at(index_of.continuators) < continuators.size());
            assert(  crossovers.size() > 0); assert(_encoding.at(index_of.crossovers)   <   crossovers.size());
            assert(   mutations.size() > 0); assert(_encoding.at(index_of.mutations)    <    mutations.size());
            assert(   selectors.size() > 0); assert(_encoding.at(index_of.selectors)    <    selectors.size());
            assert(replacements.size() > 0); assert(_encoding.at(index_of.replacements) < replacements.size());

            eoSequentialOp<EOT> variator;
            variator.add(this->crossover(), 1.0);
            variator.add(this->mutation(), 1.0);

            eoGeneralBreeder<EOT> breeder(this->selector(), variator, 1.0);

            eoGenContinue<EOT> common_cont(100);
            eoCombinedContinue<EOT> gen_cont(common_cont);
            gen_cont.add(this->continuator());

            eoEasyEA<EOT> algo = eoEasyEA<EOT>(gen_cont, _eval, breeder, this->replacement());

            algo(pop);
        }

        std::string name()
        {
            std::ostringstream name;
            name << _encoding.at(index_of.continuators) << " (" << this->continuator().className() << ") + ";
            name << _encoding.at(index_of.crossovers)   << " (" << this->crossover().className()   << ") + ";
            name << _encoding.at(index_of.mutations)    << " (" << this->mutation().className()    << ") + ";
            name << _encoding.at(index_of.selectors)    << " (" << this->selector().className()    << ") + ";
            name << _encoding.at(index_of.replacements) << " (" << this->replacement().className() << ")";
            return name.str();
        }

    protected:
        eoEvalFunc<EOT>& _eval;

        eoContinue<EOT>& continuator()
        {
            assert(_encoding.at(index_of.continuators) < continuators.size());
            return continuators.instanciate(_encoding.at(index_of.continuators));
        }

        eoQuadOp<EOT>& crossover()
        {
            assert(_encoding.at(index_of.crossovers) < crossovers.size());
            return crossovers.instanciate(_encoding.at(index_of.crossovers));
        }

        eoMonOp<EOT>& mutation()
        {
            assert(_encoding.at(index_of.mutations) < mutations.size());
            return mutations.instanciate(_encoding.at(index_of.mutations));
        }

        eoSelectOne<EOT>& selector()
        {
            assert(_encoding.at(index_of.selectors) < selectors.size());
            return selectors.instanciate(_encoding.at(index_of.selectors));
        }

        eoReplacement<EOT>& replacement()
        {
            assert(_encoding.at(index_of.replacements) < replacements.size());
            return replacements.instanciate(_encoding.at(index_of.replacements));
        }

};


int main(int /*argc*/, char** /*argv*/)
{
    size_t dim = 500;
    size_t pop_size = 10;

    eo::log << eo::setlevel(eo::warnings);

    using EOT = eoBit<double>;

    oneMaxEval<EOT> eval;

    eoBooleanGenerator gen(0.5);
    eoInitFixedLength<EOT> init(dim, gen);

    eoFoundryEA<EOT> foundry(eval);

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
    for(size_t i=2; i < 10; i+=4) {
        foundry.replacements.add< eoSSGADetTournamentReplacement<EOT> >(i);
    }
    foundry.replacements.add< eoSSGAStochTournamentReplacement<EOT> >(0.51);


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
                        apply(eval,pop);

                        foundry.at(foundry.index_of.continuators) = i_cont;
                        foundry.at(foundry.index_of.crossovers) = i_cross;
                        foundry.at(foundry.index_of.mutations) = i_mut;
                        foundry.at(foundry.index_of.selectors) = i_sel;
                        foundry.at(foundry.index_of.replacements) = i_rep;

                        // Or, if you know the order.
                        foundry = {i_cont, i_cross, i_mut, i_sel, i_rep};

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
