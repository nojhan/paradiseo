
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

#ifndef _eoEvalFoundryEA_H_
#define _eoEvalFoundryEA_H_

#include "eoEvalFunc.h"
#include "eoAlgoFoundryEA.h"
#include "eoInit.h"
#include "eoPopEvalFunc.h"

/** Evaluate an algorithm assembled by an eoAlgoFoundryEA, encoded as a numeric vector.
 *
 * Allows to plug another search algorithm on top of an eoAlgoFoundryEA,
 * so as to find the best configuration.
 *
 * The first template EOT is the encoding of the high-level algorithm selection problem,
 * the second template SUB is the encoding of the low-level generic problem.
 *
 * @ingroup Evaluation
 * @ingroup Foundry
 */
template<class EOT, class SUB>
class eoEvalFoundryEA : public eoEvalFunc<EOT>
{
public:
    /** Takes the necessary parameters to perform a search on the sub-problem.
     *
     * @param foundry The set of algorithms among which to select.
     * @param subpb_init An initilizer for sub-problem encoding.
     * @param pop_size Population size for the sub-problem solver.
     * @param subpb_eval The sub-problem itself.
     * @param penalization If any solution to the high-level algorithm selection problem is out of bounds, set it to this value.
     */
    eoEvalFoundryEA(
            eoAlgoFoundryEA<SUB>& foundry,
            eoInit<SUB>& subpb_init,
            const size_t pop_size,
            eoPopEvalFunc<SUB>& subpb_eval,
            const typename SUB::Fitness penalization
        ) :
            _subpb_init(subpb_init),
            _subpb_eval(subpb_eval),
            _foundry(foundry),
            _penalization(penalization),
            _pop_size(pop_size),
            i_cont(foundry.continuators.index()),
            i_cros(foundry.crossovers.index()),
            i_muta(foundry.mutations.index()),
            i_sele(foundry.selectors.index()),
            i_repl(foundry.replacements.index())
    { }

public:

    /** Decode the high-level problem encoding as an array of indices.
     *
     * @note: If the EOT is an eoInt, this will be optimized out.
     *
     * May be useful for getting a solution back into an eoAlgoFoundryEA.
     * @code
     * foundry = eval.decode(pop.best_element());
     * std::cout << foundry.name() << std::endl;
     * auto& cont = foundry.continuator(); // Get the configured operator
     * @encode
     */
    typename eoAlgoFoundry<SUB>::Encodings decode( const EOT& sol ) const
    {
        // // Denormalize
        // size_t cont = static_cast<size_t>(std::ceil( sol[i_cont] * _foundry.continuators.size() ));
        // size_t cros = static_cast<size_t>(std::ceil( sol[i_cros] * _foundry.crossovers  .size() ));
        // size_t muta = static_cast<size_t>(std::ceil( sol[i_muta] * _foundry.mutations   .size() ));
        // size_t sele = static_cast<size_t>(std::ceil( sol[i_sele] * _foundry.selectors   .size() ));
        // size_t repl = static_cast<size_t>(std::ceil( sol[i_repl] * _foundry.replacements.size() ));

        // Direct encoding
        size_t cont = static_cast<size_t>(std::ceil( sol[i_cont] ));
        size_t cros = static_cast<size_t>(std::ceil( sol[i_cros] ));
        size_t muta = static_cast<size_t>(std::ceil( sol[i_muta] ));
        size_t sele = static_cast<size_t>(std::ceil( sol[i_sele] ));
        size_t repl = static_cast<size_t>(std::ceil( sol[i_repl] ));

        return {cont, cros, muta, sele, repl};
    }

    /** Perform a sub-problem search with the configuration encoded in the given solution
     *  and set its (high-level) fitness to the best (low-level) fitness found.
     *
     * You may want to overload this to perform multiple runs or solve multiple sub-problems.
     */
    virtual void operator()(EOT& sol)
    {
        if(not sol.invalid()) {
            return;
        }

        eoPop<SUB> pop;
        pop.append(_pop_size, _subpb_init);
        _subpb_eval(pop,pop);

        auto config = decode(sol);
        size_t cont = std::get<size_t>(config[i_cont]);
        size_t cros = std::get<size_t>(config[i_cros]);
        size_t muta = std::get<size_t>(config[i_muta]);
        size_t sele = std::get<size_t>(config[i_sele]);
        size_t repl = std::get<size_t>(config[i_repl]);

        if(
               cont < _foundry.continuators.size()
           and cros < _foundry.crossovers  .size()
           and muta < _foundry.mutations   .size()
           and sele < _foundry.selectors   .size()
           and repl < _foundry.replacements.size()
        ) {
            _foundry.select(config);

            // Actually perform a search
            _foundry(pop);

            sol.fitness( pop.best_element().fitness() );
        } else {
            sol.fitness( _penalization ); // penalization
        }
    }

protected:
    eoInit<SUB>& _subpb_init;
    eoPopEvalFunc<SUB>& _subpb_eval;
    eoAlgoFoundryEA<SUB>& _foundry;
    const typename EOT::Fitness _penalization;
    const size_t _pop_size;

    const size_t i_cont;
    const size_t i_cros;
    const size_t i_muta;
    const size_t i_sele;
    const size_t i_repl;

};

/** Helper function to instanciate an eoEvalFoundryEA without having to indicate the template for the sub-problem encoding.
 *
 * The template is deduced from the constructor's parameters.
 * Not sure it's more concise than a classical instanciation…
 */
template<class EOT, class SUB>
eoEvalFoundryEA<EOT,SUB>&
    make_eoEvalFoundryEA(
        eoInit<SUB>& subpb_init,
        eoPopEvalFunc<SUB>& subpb_eval,
        eoAlgoFoundryEA<SUB>& foundry,
        const typename SUB::Fitness penalization,
        const size_t pop_size
    )
{
    return *(new eoEvalFoundryEA<EOT,SUB>(subpb_init, subpb_eval, foundry, penalization, pop_size));
}

#endif // _eoEvalFoundryEA_H_

