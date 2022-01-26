
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
   © 2022 Institut Pasteur

    Authors:
        Johann Dreo <johann@dreo.fr>
*/

#ifndef _eoAlgoFoundryEA_H_
#define _eoAlgoFoundryEA_H_

#include <array>
#include <tuple>

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
 *   foundry.selectors.add< eoStochTournamentSelect<EOT> >( 0.5 );
 * @endcode
 *
 * @warning If the constructor takes a reference YOU SHOULD ABSOLUTELY wrap it
 * in a `std::ref`, or it will silently be passed as a copy,
 * which would effectively disable any link between operators.
 *
 * In a second step, the operators to be used should be selected
 * by indicating their index, just like the foundry was a array of five elements:
 * @code
 * foundry = {size_t{0}, size_t{1}, size_t{2}, size_t{0}, size_t{3}};
 * //                ^          ^          ^          ^          ^ replacement
 * //                |          |          |          + selection
 * //                |          |          + mutation
 * //                |          + crossover
 * //                + continue
 * @endcode
 *
 * @note: by default, the firsts of the five operators are selected.
 *
 * If you don't (want to) recall the order of the operators in the encoding,
 * you can use the `index()` member, for example:
 * @code
 *   foundry.at(foundry.continuators.index()) = size_t{2}; // select the third continuator
 * @endcode
 *
 * Now, you can call the fourdry just like any eoAlgo, by passing it an eoPop:
 * @code
 *   foundry(pop);
 * @encode
 * It will instantiate the needed operators (only) and the algorithm itself on-the-fly,
 * and then run it.
 *
 * @note: Thanks to the underlying eoOperatorFoundry, not all the added operators are instantiated.
 * Every instantiation is deferred upon actual use. That way, you can still reconfigure them
 * at any time with `eoForgeOperator::setup`, for example:
 * @code
 *   foundry.selectors.at(0).setup(0.5); // using constructor's arguments
 * @endcode
 *
 * @ingroup Foundry
 * @ingroup Algorithms
 */
template<class EOT>
class eoAlgoFoundryEA : public eoAlgoFoundry<EOT>
{
    public:
        /** The constructon only take an eval, because all other operators
         * are stored in the public containers.
         */
        eoAlgoFoundryEA( eoPopEvalFunc<EOT>& eval, size_t max_gen ) :
            eoAlgoFoundry<EOT>(5),
            continuators(0, true), // Always re-instantiate continuators, because they hold a state.
            crossovers(1, false),
            mutations(2, false),
            selectors(3, false),
            replacements(4, false),
            _eval(eval),
            _max_gen(max_gen)
        { }

    public:

        /* Operators containers @{ */
        eoOperatorFoundry< eoContinue<EOT>    > continuators;
        eoOperatorFoundry< eoQuadOp<EOT>      > crossovers;
        eoOperatorFoundry< eoMonOp<EOT>       > mutations;
        eoOperatorFoundry< eoSelectOne<EOT>   > selectors;
        eoOperatorFoundry< eoReplacement<EOT> > replacements;
        /* @} */

        /** instantiate and call the pre-selected algorithm.
         */
        void operator()(eoPop<EOT>& pop)
        {
            assert(continuators.size() > 0); assert(this->rank(continuators) < continuators.size());
            assert(  crossovers.size() > 0); assert(this->rank(  crossovers) <   crossovers.size());
            assert(   mutations.size() > 0); assert(this->rank(   mutations) <    mutations.size());
            assert(   selectors.size() > 0); assert(this->rank(   selectors) <    selectors.size());
            assert(replacements.size() > 0); assert(this->rank(replacements) < replacements.size());

            eoSequentialOp<EOT> variator;
            variator.add(this->crossover(), 1.0);
            variator.add(this->mutation(), 1.0);

            eoGeneralBreeder<EOT> breeder(this->selector(), variator, 1.0);

            eoGenContinue<EOT> common_cont(_max_gen);
            eoCombinedContinue<EOT> gen_cont(common_cont);
            gen_cont.add(this->continuator());

            eoEasyEA<EOT> algo = eoEasyEA<EOT>(gen_cont, _eval, breeder, this->replacement());

            algo(pop);
        }

        /** Return an approximate name of the seected algorithm.
         *
         * @note: does not take into account parameters of the operators,
         * only show class names.
         */
        std::string name()
        {
            std::ostringstream name;
            name << this->continuator().className() << " [" << this->rank(continuators) << "] + ";
            name << this->crossover()  .className() << " [" << this->rank(crossovers)   << "] + ";
            name << this->mutation()   .className() << " [" << this->rank(mutations)    << "] + ";
            name << this->selector()   .className() << " [" << this->rank(selectors)    << "] + ";
            name << this->replacement().className() << " [" << this->rank(replacements) << "]";
            return name.str();
        }

    protected:
        eoPopEvalFunc<EOT>& _eval;
        const size_t _max_gen;

    public:
        /** Currently selected continuator.
         */
        eoContinue<EOT>& continuator()
        {
            assert(this->rank(continuators) < continuators.size());
            return continuators.instantiate(this->rank(continuators));
        }

        /** Currently selected crossover.
         */
        eoQuadOp<EOT>& crossover()
        {
            assert(this->rank(crossovers) < crossovers.size());
            return crossovers.instantiate(this->rank(crossovers));
        }

        /** Currently selected mutation.
         */
        eoMonOp<EOT>& mutation()
        {
            assert(this->rank(mutations) < mutations.size());
            return mutations.instantiate(this->rank(mutations));
        }

        /** Currently selected selector.
         */
        eoSelectOne<EOT>& selector()
        {
            assert(this->rank(selectors) < selectors.size());
            return selectors.instantiate(this->rank(selectors));
        }

        /** Currently selected replacement.
         */
        eoReplacement<EOT>& replacement()
        {
            assert(this->rank(replacements) < replacements.size());
            return replacements.instantiate(this->rank(replacements));
        }

};

#endif // _eoAlgoFoundryEA_H_
