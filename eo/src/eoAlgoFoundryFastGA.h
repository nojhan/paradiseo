
/*
   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation;
   version 2 of the License.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

   © 2020 Thales group

    Authors:
        Johann Dreo <johann.dreo@thalesgroup.com>
*/

#ifndef _eoAlgoFoundryFastGA_H_
#define _eoAlgoFoundryFastGA_H_

#include <array>
#include <tuple>
#include <limits>

/** A class that assemble an eoEasyEA on the fly, given a combination of available operators.
 *
 * The foundry should first be set up with sets of operators
 * for the main modules of an EA:
 * continuators, crossovers, mutations, selection and replacement operators.
 *
 * This is done through public member variable's `add` method,
 * which takes the class name as template and its constructor's parameters
 * as arguments. For example:
 * @code
 * foundry.selectors.add< eoStochTournamentSelect<EOT> >( 0.5 );
 * @endcode
 *
 * @warning If the constructor takes a reference YOU SHOULD ABSOLUTELY wrap it
 * in a `std::ref`, or it will silently be passed as a copy,
 * which would effectively disable any link between operators.
 *
 * In a second step, the operators to be used should be selected
 * by indicating their index, passing an array of eight elements:
 * @code
 * foundry.select({0, 1, 2, 3, 4, 5, 6, 7});
 * @endcode
 *
 * @note: by default, the firsts of the eight operators are selected.
 *
 * If you don't (want to) recall the order of the operators in the encoding,
 * you can use the `index()` member, for example:
 * @code
 * foundry.at(foundry.continuators.index()) = 2; // select the third continuator
 * @endcode
 *
 * Now, you can call the fourdry just like any eoAlgo, by passing it an eoPop:
 * @code
 * foundry(pop);
 * @encode
 * It will instantiate the needed operators (only) and the algorithm itself on-the-fly,
 * and then run it.
 *
 * @note: Thanks to the underlying eoOperatorFoundry, not all the added operators are instantiated.
 * Every instantiation is deferred upon actual use. That way, you can still reconfigure them
 * at any time with `eoForgeOperator::setup`, for example:
 * @code
 * foundry.selector.at(0).setup(0.5); // using constructor's arguments
 * @endcode
 *
 * @ingroup Foundry
 * @ingroup Algorithms
 */
template<class EOT>
class eoAlgoFoundryFastGA : public eoAlgoFoundry<EOT>
{
    public:
        /** The constructon only take an eval, because all other operators
         * are stored in the public containers.
         */
        eoAlgoFoundryFastGA( eoInit<EOT> & init, eoEvalFunc<EOT>& eval, size_t max_evals = 10000, size_t max_restarts = std::numeric_limits<size_t>::max() ) :
            eoAlgoFoundry<EOT>(8),
            continuators(0, true), // Always re-instantiate continuators, because they hold a state.
            crossover_rates(1, false),
            crossovers(2, false),
            mutation_rates(3, false),
            mutations(4, false),
            selectors(5, false),
            pop_sizes(6, false),
            replacements(7, false),
            _eval(eval),
            _init(init),
            _max_evals(max_evals),
            _max_restarts(max_restarts)
        { }

    public:

        /* Operators containers @{ */
        eoOperatorFoundry< eoContinue<EOT>    > continuators;
        eoOperatorFoundry< double             > crossover_rates;
        eoOperatorFoundry< eoQuadOp<EOT>      > crossovers;
        eoOperatorFoundry< double             > mutation_rates;
        eoOperatorFoundry< eoMonOp<EOT>       > mutations;
        eoOperatorFoundry< eoSelectOne<EOT>   > selectors;
        eoOperatorFoundry< size_t             > pop_sizes;
        eoOperatorFoundry< eoReplacement<EOT> > replacements;
        /* @} */

        /** instantiate and call the pre-selected algorithm.
         */
        void operator()(eoPop<EOT>& pop)
        {
            assert(continuators.size() > 0); assert(this->at(continuators.index()) < continuators.size());
            assert(  crossover_rates.size() > 0); assert(this->at(  crossover_rates.index()) <   crossover_rates.size());
            assert(  crossovers.size() > 0); assert(this->at(  crossovers.index()) <   crossovers.size());
            assert(   mutation_rates.size() > 0); assert(this->at(   mutation_rates.index()) <    mutation_rates.size());
            assert(   mutations.size() > 0); assert(this->at(   mutations.index()) <    mutations.size());
            assert(   selectors.size() > 0); assert(this->at(   selectors.index()) <    selectors.size());
            assert(   pop_sizes.size() > 0); assert(this->at(   pop_sizes.index()) <    pop_sizes.size());
            assert(replacements.size() > 0); assert(this->at(replacements.index()) < replacements.size());

            // Crossover or clone
            double cross_rate = this->crossover_rate();
            eoProportionalOp<EOT> cross;
            // Cross-over that produce only one offspring,
            // made by wrapping the quad op (which produce 2 offsprings)
            // in a bin op (which ignore the second offspring).
            eoQuad2BinOp<EOT> single_cross(this->crossover());
            cross.add(single_cross, cross_rate);
            eoBinCloneOp<EOT> cross_clone;
            cross.add(cross_clone, 1 - cross_rate); // Clone

            // Mutation or clone
            double mut_rate = this->mutation_rate();
            eoProportionalOp<EOT> mut;
            mut.add(this->mutation(), mut_rate);
            eoMonCloneOp<EOT> mut_clone;
            mut.add(mut_clone, 1 - mut_rate); // FIXME TBC

            // Apply mutation after cross-over.
            eoSequentialOp<EOT> variator;
            variator.add(cross,1.0);
            variator.add(mut,1.0);

            // All variatiors
            double lambda = this->pop_size();
            eoGeneralBreeder<EOT> breeder(this->selector(), variator, lambda, /*as rate*/false);

            // Objective function calls counter
            eoEvalCounterThrowException<EOT> eval(_eval, _max_evals);
            eoPopLoopEval<EOT> pop_eval(eval);

            // Algorithm itself
            eoEasyEA<EOT> algo = eoEasyEA<EOT>(this->continuator(), pop_eval, breeder, this->replacement());

            // Restart wrapper
            eoAlgoPopReset<EOT> reset_pop(_init, pop_eval);
            eoGenContinue<EOT> restart_cont(_max_restarts);
            eoAlgoRestart<EOT> restart(eval, algo, restart_cont, reset_pop);

            try {
                restart(pop);
            } catch(eoMaxEvalException e) {
                // In case some solutions were not evaluated when max eval occured.
                eoPopLoopEval<EOT> pop_last_eval(_eval);
                pop_last_eval(pop,pop);
            }
        }

        /** Return an approximate name of the selected algorithm.
         *
         * @note: does not take into account parameters of the operators,
         * only show class names.
         */
        std::string name()
        {
            std::ostringstream name;
            name << this->at(continuators.index()) << " (" << this->continuator().className() << ") + ";
            name << this->at(crossover_rates.index())   << " (" << this->crossover_rate().className()   << ") + ";
            name << this->at(crossovers.index())   << " (" << this->crossover().className()   << ") + ";
            name << this->at(mutation_rates.index())    << " (" << this->mutation_rate().className()    << ") + ";
            name << this->at(mutations.index())    << " (" << this->mutation().className()    << ") + ";
            name << this->at(selectors.index())    << " (" << this->selector().className()    << ") + ";
            name << this->at(pop_sizes.index()) << " (" << this->pop_size().className() << ")";
            name << this->at(replacements.index()) << " (" << this->replacement().className() << ")";
            return name.str();
        }

    protected:
        eoEvalFunc<EOT>& _eval;
        eoInit<EOT>& _init;
        const size_t _max_evals;
        const size_t _max_restarts;

    public:
        eoContinue<EOT>& continuator()
        {
            assert(this->at(continuators.index()) < continuators.size());
            return continuators.instantiate(this->at(continuators.index()));
        }

        double& crossover_rate()
        {
            assert(this->at(crossover_rates.index()) < crossover_rates.size());
            return crossover_rates.instantiate(this->at(crossover_rates.index()));
        }

        eoQuadOp<EOT>& crossover()
        {
            assert(this->at(crossovers.index()) < crossovers.size());
            return crossovers.instantiate(this->at(crossovers.index()));
        }

        double& mutation_rate()
        {
            assert(this->at(mutation_rates.index()) < mutation_rates.size());
            return mutation_rates.instantiate(this->at(mutation_rates.index()));
        }

        eoMonOp<EOT>& mutation()
        {
            assert(this->at(mutations.index()) < mutations.size());
            return mutations.instantiate(this->at(mutations.index()));
        }

        eoSelectOne<EOT>& selector()
        {
            assert(this->at(selectors.index()) < selectors.size());
            return selectors.instantiate(this->at(selectors.index()));
        }

        size_t& pop_size()
        {
            assert(this->at(pop_sizes.index()) < pop_sizes.size());
            return pop_sizes.instantiate(this->at(pop_sizes.index()));
        }

        eoReplacement<EOT>& replacement()
        {
            assert(this->at(replacements.index()) < replacements.size());
            return replacements.instantiate(this->at(replacements.index()));
        }

};

#endif // _eoAlgoFoundryFastGA_H_
