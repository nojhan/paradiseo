
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
 * The foundry should first be set up with sets of operators/parameters
 * for the main modules of a FastGA:
 * continuators, crossovers (and rate of call), mutations (and rate of call),
 * selections, replacement operators, offspring size, etc.
 *
 * This is done through public member variable's `add` method,
 * which takes the class name as template and its constructor's parameters
 * as arguments. For example:
 * @code
 *   foundry.selectors.add< eoRandomSelect<EOT> >();
 * @endcode
 *
 * @warning If the constructor takes a reference YOU SHOULD ABSOLUTELY wrap it
 *          in a `std::ref`, or it will silently be passed as a copy,
 *          which would effectively disable any link with other operator(s).
 *
 * In a second step, the operators to be used should be selected
 * by indicating their wanted index or value, passing an array of 10 elements:
 * @code
 *   foundry.select({
 *       double{0.1}, // crossover rate
 *       size_t{1},   // crossover selector
 *       size_t{2},   // crossover
 *       size_t{3},   // selector after crossover
 *       double{0.4}, // mutation rate
 *       size_t{5},   // mutation selector
 *       size_t{6},   // mutation
 *       size_t{7},   // replacement
 *       size_t{8},   // continuator
 *       size_t{9}    // nb of offsprings
 *   });
 * @endcode
 *
 * @note: by default, the firsts of the 10 operators are selected.
 *
 * If you don't (want to) recall the order of the operators in the encoding,
 * you can use the `index()` member, for example:
 * @code
 *   foundry.at(foundry.continuators.index()) = size_t{2}; // select the third continuator
 * @endcode
 *
 * Now, you can call the foundry just like any eoAlgo, by passing it an eoPop:
 * @code
 *   foundry(pop);
 * @encode
 * It will instantiate the needed operators (only) and the algorithm itself on-the-fly,
 * and then run it.
 *
 * @note: Thanks to the underlying eoOperatorFoundry, not all the added operators are instantiated.
 *        Every instantiation is deferred upon actual use. That way, you can still reconfigure them
 *        at any time with `eoForgeOperator::setup`, for example:
 *        @code
 *          foundry.selector.at(0).setup(0.5); // Will call constructor's arguments
 *        @endcode
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
                const size_t max_evals = 10000,
                const size_t max_restarts = std::numeric_limits<size_t>::max()
            ) :
            eoAlgoFoundry<EOT>(10),

            crossover_rates(0, 0.0, 1.0),
            crossover_selectors(1, false),
            crossovers(2, false),
            aftercross_selectors(3, false),

            mutation_rates(4, 0.0, 1.0),
            mutation_selectors(5, false),
            mutations(6, false),

            replacements(7, false),
            continuators(8, true), // Always re-instantiate continuators, because they hold a state.
            offspring_sizes(9, 0, std::numeric_limits<size_t>::max()),
            _eval(eval),
            _init(init),
            _max_evals(max_evals),
            _max_restarts(max_restarts)
        { }

    public:

        /* Operators containers @{ */
        eoParameterFoundry< double            > crossover_rates;
        eoOperatorFoundry< eoSelectOne<EOT>   > crossover_selectors;
        eoOperatorFoundry< eoQuadOp<EOT>      > crossovers;
        eoOperatorFoundry< eoSelectOne<EOT>   > aftercross_selectors;

        eoParameterFoundry< double            > mutation_rates;
        eoOperatorFoundry< eoSelectOne<EOT>   > mutation_selectors;
        eoOperatorFoundry< eoMonOp<EOT>       > mutations;

        eoOperatorFoundry< eoReplacement<EOT> > replacements;
        eoOperatorFoundry< eoContinue<EOT>    > continuators;
        eoParameterFoundry< size_t            > offspring_sizes;
        /* @} */

        /** instantiate and call the pre-selected algorithm.
         */
        void operator()(eoPop<EOT>& pop)
        {
            assert( crossover_selectors.size() > 0); assert(this->rank( crossover_selectors) <  crossover_selectors.size());
            assert(          crossovers.size() > 0); assert(this->rank(          crossovers) <           crossovers.size());
            assert(aftercross_selectors.size() > 0); assert(this->rank(aftercross_selectors) < aftercross_selectors.size());
            assert(  mutation_selectors.size() > 0); assert(this->rank(  mutation_selectors) <   mutation_selectors.size());
            assert(           mutations.size() > 0); assert(this->rank(           mutations) <            mutations.size());
            assert(        replacements.size() > 0); assert(this->rank(        replacements) <         replacements.size());
            assert(        continuators.size() > 0); assert(this->rank(        continuators) <         continuators.size());

            // Objective function calls counter
            eoEvalCounterThrowException<EOT> eval(_eval, _max_evals);
            eo::log << eo::xdebug << "Evaluations: " << eval.value() << " / " << _max_evals << std::endl;
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
            // eoAlgoPopReset<EOT> reset_pop(_init, pop_eval);
            // eoGenContinue<EOT> restart_cont(_max_restarts);
            // eoAlgoRestart<EOT> restart(eval, algo, restart_cont, reset_pop);

            try {
                // restart(pop);
                algo(pop);
            } catch(eoMaxEvalException & e) {
#ifndef NDEBUG
                eo::log << eo::debug << "Reached maximum evaluations: " << eval.getValue() << " / " << _max_evals << std::endl;
#endif
                // In case some solutions were not evaluated when max eval occured.
                // FIXME can this even be considered legal?
                // eoPopLoopEval<EOT> pop_last_eval(_eval);
                // pop_last_eval(pop,pop);
            }
        }

        /** Return an approximate name of the selected algorithm.
         */
        std::string name()
        {
            std::ostringstream name;
            name << "crossover_rate: "      << this->     crossover_rate()             << " + ";
            name << "crossover_selector: "  << this-> crossover_selector().className() << " [" << this->rank( crossover_selectors) << "] + ";
            name << "aftercross_selector: " << this->aftercross_selector().className() << " [" << this->rank(aftercross_selectors) << "] + ";
            name << "crossover: "           << this->          crossover().className() << " [" << this->rank(          crossovers) << "] + ";
            name << "mutation_rate: "       << this->      mutation_rate()             << " + ";
            name << "mutation_selector: "   << this->  mutation_selector().className() << " [" << this->rank(  mutation_selectors) << "] + ";
            name << "mutation: "            << this->           mutation().className() << " [" << this->rank(           mutations) << "] + ";
            name << "replacement: "         << this->        replacement().className() << " [" << this->rank(        replacements) << "] + ";
            name << "continuator: "         << this->        continuator().className() << " [" << this->rank(        continuators) << "] + ";
            name << "offspring_size: "      << this->     offspring_size()             << "";

           return name.str();
        }

    protected:
        eoEvalFunc<EOT>& _eval;
        eoInit<EOT>& _init;
        const size_t _max_evals;
        const size_t _max_restarts;

    public:
        /** Currently selected continuator.
         */
        eoContinue<EOT>& continuator()
        {
            const size_t r = this->rank(continuators);
            assert(r < continuators.size());
            return continuators.instantiate(r);
        }

        /** Currently selected crossover_rate.
         */
        double& crossover_rate()
        {
            // We could have used `decltype(crossover_rates)::Type` instead of `double`, here,
            // but this is less readable and the type is declared just above,
            // so we are supposed to know it.
            const double val = this->value(crossover_rates);
            assert(crossover_rates.min() <= val and val <= crossover_rates.max());
            return crossover_rates.instantiate(val);
        }

        /** Currently selected crossover.
         */
        eoQuadOp<EOT>& crossover()
        {
            const size_t r = this->rank(crossovers);
            assert(r < crossovers.size());
            return crossovers.instantiate(r);
        }

        /** Currently selected mutation_rate.
         */
        double& mutation_rate()
        {
            const double val = this->value(mutation_rates);
            assert(mutation_rates.min() <= val and val <= mutation_rates.max());
            return mutation_rates.instantiate(val);
        }

        /** Currently selected mutation.
         */
        eoMonOp<EOT>& mutation()
        {
            const size_t r = this->rank(mutations);
            assert(r < mutations.size());
            return mutations.instantiate(r);
        }

        /** Currently selected crossover_selector.
         */
        eoSelectOne<EOT>& crossover_selector()
        {
            const size_t r = this->rank(crossover_selectors);
            assert(r < crossover_selectors.size());
            return crossover_selectors.instantiate(r);
        }

        /** Currently selected aftercross_selector.
         */
        eoSelectOne<EOT>& aftercross_selector()
        {
            const size_t r = this->rank(aftercross_selectors);
            assert(r < aftercross_selectors.size());
            return aftercross_selectors.instantiate(r);
        }

        /** Currently selected mutation_selector.
         */
        eoSelectOne<EOT>& mutation_selector()
        {
            const size_t r = this->rank(mutation_selectors);
            assert(r < mutation_selectors.size());
            return mutation_selectors.instantiate(r);
        }

        /** Currently selected offspring_size.
         */
        size_t& offspring_size()
        {
            const size_t val = this->len(offspring_sizes);
            assert(offspring_sizes.min() <= val and val <= offspring_sizes.max());
            return offspring_sizes.instantiate(val);
        }

        /** Currently selected replacement.
         */
        eoReplacement<EOT>& replacement()
        {
            const size_t r = this->rank(replacements);
            assert(r < replacements.size());
            return replacements.instantiate(r);
        }

};

#endif // _eoAlgoFoundryFastGA_H_
