
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

/** A class that assemble an eoFastGA on the fly, given a combination of available operators.
 *
 * The foundry should first be set up with sets of operators
 * for the main modules of a FastGA:
 * continuators, crossovers, mutations, selections, replacement operators, etc.
 *
 * This is done through public member variable's `add` method,
 * which takes the class name as template and its constructor's parameters
 * as arguments. For example:
 * @code
 * foundry.selectors.add< eoRandomSelect<EOT> >();
 * @endcode
 *
 * @warning If the constructor takes a reference YOU SHOULD ABSOLUTELY wrap it
 * in a `std::ref`, or it will silently be passed as a copy,
 * which would effectively disable any link with other operator(s).
 *
 * In a second step, the operators to be used should be selected
 * by indicating their index, passing an array of 10 elements:
 * @code
 * foundry.select({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
 * @endcode
 *
 * @note: by default, the firsts of the 10 operators are selected.
 *
 * If you don't (want to) recall the order of the operators in the encoding,
 * you can use the `index()` member, for example:
 * @code
 * foundry.at(foundry.continuators.index()) = 2; // select the third continuator
 * @endcode
 *
 * Now, you can call the foundry just like any eoAlgo, by passing it an eoPop:
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
 * foundry.selector.at(0).setup(0.5); // Will call constructor's arguments
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
        eoAlgoFoundryFastGA(
                eoInit<EOT> & init,
                eoEvalFunc<EOT>& eval,
                size_t max_evals = 10000,
                size_t max_restarts = std::numeric_limits<size_t>::max()
            ) :
            eoAlgoFoundry<EOT>(10),

            crossover_rates(0, false),
            crossover_selectors(1, false),
            crossovers(2, false),
            aftercross_selectors(3, false),

            mutation_rates(4, false),
            mutation_selectors(5, false),
            mutations(6, false),

            replacements(7, false),
            continuators(8, true), // Always re-instantiate continuators, because they hold a state.
            offspring_sizes(9, false),
            _eval(eval),
            _init(init),
            _max_evals(max_evals),
            _max_restarts(max_restarts)
        { }

    public:

        /* Operators containers @{ */
        eoOperatorFoundry< double             > crossover_rates;
        eoOperatorFoundry< eoSelectOne<EOT>   > crossover_selectors;
        eoOperatorFoundry< eoQuadOp<EOT>      > crossovers;
        eoOperatorFoundry< eoSelectOne<EOT>   > aftercross_selectors;

        eoOperatorFoundry< double             > mutation_rates;
        eoOperatorFoundry< eoSelectOne<EOT>   > mutation_selectors;
        eoOperatorFoundry< eoMonOp<EOT>       > mutations;

        eoOperatorFoundry< eoReplacement<EOT> > replacements;
        eoOperatorFoundry< eoContinue<EOT>    > continuators;
        eoOperatorFoundry< size_t             > offspring_sizes;
        /* @} */

        /** instantiate and call the pre-selected algorithm.
         */
        void operator()(eoPop<EOT>& pop)
        {
            assert(     crossover_rates.size() > 0); assert(this->at(     crossover_rates.index()) <      crossover_rates.size());
            assert( crossover_selectors.size() > 0); assert(this->at( crossover_selectors.index()) <  crossover_selectors.size());
            assert(          crossovers.size() > 0); assert(this->at(          crossovers.index()) <           crossovers.size());
            assert(aftercross_selectors.size() > 0); assert(this->at(aftercross_selectors.index()) < aftercross_selectors.size());
            assert(      mutation_rates.size() > 0); assert(this->at(      mutation_rates.index()) <       mutation_rates.size());
            assert(  mutation_selectors.size() > 0); assert(this->at(  mutation_selectors.index()) <   mutation_selectors.size());
            assert(           mutations.size() > 0); assert(this->at(           mutations.index()) <            mutations.size());
            assert(        replacements.size() > 0); assert(this->at(        replacements.index()) <         replacements.size());
            assert(        continuators.size() > 0); assert(this->at(        continuators.index()) <         continuators.size());
            assert(           offspring_sizes.size() > 0); assert(this->at(           offspring_sizes.index()) <            offspring_sizes.size());

            // Objective function calls counter
            eoEvalCounterThrowException<EOT> eval(_eval, _max_evals);
            eoPopLoopEval<EOT> pop_eval(eval);

            // Algorithm itself
            eoFastGA<EOT> algo(
                this->crossover_rate(),
                this->crossover_selector(),
                this->crossover(),
                this->aftercross_selector(),
                this->mutation_rate(),
                this->mutation_selector(),
                this->mutation(),
                pop_eval,
                this->replacement(),
                this->continuator(),
                this->offspring_size()
            );

            // Restart wrapper
            eoAlgoPopReset<EOT> reset_pop(_init, pop_eval);
            eoGenContinue<EOT> restart_cont(_max_restarts);
            eoAlgoRestart<EOT> restart(eval, algo, restart_cont, reset_pop);

            try {
                restart(pop);
            } catch(eoMaxEvalException e) {
                // In case some solutions were not evaluated when max eval occured.
                // FIXME can this even be considered legal?
                eoPopLoopEval<EOT> pop_last_eval(_eval);
                pop_last_eval(pop,pop);
            }
        }

        /** Return an approximate name of the selected algorithm.
         */
        std::string name()
        {
            std::ostringstream name;
            name << this->at(     crossover_rates.index()) << " (" << this->     crossover_rate()             << ") + ";
            name << this->at( crossover_selectors.index()) << " (" << this-> crossover_selector().className() << ") + ";
            name << this->at(aftercross_selectors.index()) << " (" << this->aftercross_selector().className() << ") + ";
            name << this->at(          crossovers.index()) << " (" << this->          crossover().className() << ") + ";
            name << this->at(      mutation_rates.index()) << " (" << this->      mutation_rate()             << ") + ";
            name << this->at(  mutation_selectors.index()) << " (" << this->  mutation_selector().className() << ") + ";
            name << this->at(           mutations.index()) << " (" << this->           mutation().className() << ") + ";
            name << this->at(        replacements.index()) << " (" << this->        replacement().className() << ") + ";
            name << this->at(        continuators.index()) << " (" << this->        continuator().className() << ") + ";
            name << this->at(           offspring_sizes.index()) << " (" << this->           offspring_size()             << ")";
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

        eoSelectOne<EOT>& crossover_selector()
        {
            assert(this->at(crossover_selectors.index()) < crossover_selectors.size());
            return crossover_selectors.instantiate(this->at(crossover_selectors.index()));
        }

        eoSelectOne<EOT>& aftercross_selector()
        {
            assert(this->at(aftercross_selectors.index()) < aftercross_selectors.size());
            return aftercross_selectors.instantiate(this->at(aftercross_selectors.index()));
        }

        eoSelectOne<EOT>& mutation_selector()
        {
            assert(this->at(mutation_selectors.index()) < mutation_selectors.size());
            return mutation_selectors.instantiate(this->at(mutation_selectors.index()));
        }

        size_t& offspring_size()
        {
            assert(this->at(offspring_sizes.index()) < offspring_sizes.size());
            return offspring_sizes.instantiate(this->at(offspring_sizes.index()));
        }

        eoReplacement<EOT>& replacement()
        {
            assert(this->at(replacements.index()) < replacements.size());
            return replacements.instantiate(this->at(replacements.index()));
        }

};

#endif // _eoAlgoFoundryFastGA_H_
