
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

#ifndef _eoEvalFoundryFastGA_H_
#define _eoEvalFoundryFastGA_H_

#include "eoEvalFunc.h"
#include "eoAlgoFoundryFastGA.h"
#include "eoInit.h"
#include "eoPopEvalFunc.h"

/** Evaluate an algorithm assembled by an eoAlgoFoundryFastGA, encoded as a numeric vector.
 *
 * Allows to plug another search algorithm on top of an eoAlgoFoundryFastGA,
 * so as to find the best configuration.
 *
 * The first template EOT is the encoding of the high-level algorithm selection problem,
 * the second template SUB is the encoding of the low-level generic problem.
 *
 * @ingroup Evaluation
 * @ingroup Foundry
 */
template<class EOT, class SUB>
class eoEvalFoundryFastGA : public eoEvalFunc<EOT>
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
    eoEvalFoundryFastGA(
            eoAlgoFoundryFastGA<SUB>& foundry,
            eoInit<SUB>& subpb_init,
            eoPopEvalFunc<SUB>& subpb_eval,
            const typename SUB::Fitness penalization
        ) :
            _subpb_init(subpb_init),
            _subpb_eval(subpb_eval),
            _foundry(foundry),
            _penalization(penalization),
            i_cont(foundry.continuators.index()),
            i_crat(foundry.crossover_rates.index()),
            i_cros(foundry.crossovers.index()),
            i_mrat(foundry.mutation_rates.index()),
            i_muta(foundry.mutations.index()),
            i_sele(foundry.selectors.index()),
            i_pops(foundry.pop_sizes.index()),
            i_repl(foundry.replacements.index())
    { }

protected:
    const size_t i_cont;
    const size_t i_crat;
    const size_t i_cros;
    const size_t i_mrat;
    const size_t i_muta;
    const size_t i_sele;
    const size_t i_pops;
    const size_t i_repl;

public:

    /** Decode the high-level problem encoding as an array of indices.
     *
     * @note: If the EOT is an eoInt, this will be optimized out.
     *
     * May be useful for getting a solution back into an eoAlgoFoundryFastGA.
     * @code
     * foundry = eval.decode(pop.best_element());
     * std::cout << foundry.name() << std::endl;
     * auto& cont = foundry.continuator(); // Get the configured operator
     * @encode
     */
    std::vector<size_t> decode( const EOT& sol ) const
    {
        // Denormalize
        // size_t cont = static_cast<size_t>(std::ceil( sol[i_cont] * _foundry.continuators   .size() ));
        // size_t crat = static_cast<size_t>(std::ceil( sol[i_crat] * _foundry.crossover_rates.size() ));
        // size_t cros = static_cast<size_t>(std::ceil( sol[i_cros] * _foundry.crossovers     .size() ));
        // size_t mrat = static_cast<size_t>(std::ceil( sol[i_mrat] * _foundry.mutation_rates .size() ));
        // size_t muta = static_cast<size_t>(std::ceil( sol[i_muta] * _foundry.mutations      .size() ));
        // size_t sele = static_cast<size_t>(std::ceil( sol[i_sele] * _foundry.selectors      .size() ));
        // size_t pops = static_cast<size_t>(std::ceil( sol[i_pops] * _foundry.pop_sizes      .size() ));
        // size_t repl = static_cast<size_t>(std::ceil( sol[i_repl] * _foundry.replacements   .size() ));

        // Direct encoding
        size_t cont = static_cast<size_t>(std::ceil( sol[i_cont] ));
        size_t crat = static_cast<size_t>(std::ceil( sol[i_crat] ));
        size_t cros = static_cast<size_t>(std::ceil( sol[i_cros] ));
        size_t mrat = static_cast<size_t>(std::ceil( sol[i_mrat] ));
        size_t muta = static_cast<size_t>(std::ceil( sol[i_muta] ));
        size_t sele = static_cast<size_t>(std::ceil( sol[i_sele] ));
        size_t pops = static_cast<size_t>(std::ceil( sol[i_pops] ));
        size_t repl = static_cast<size_t>(std::ceil( sol[i_repl] ));

        return {cont, crat, cros, mrat, muta, sele, pops, repl};
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

        auto config = decode(sol);
        double cont = config[i_cont];
        double crat = config[i_crat];
        double cros = config[i_cros];
        double mrat = config[i_mrat];
        double muta = config[i_muta];
        double sele = config[i_sele];
        double pops = config[i_pops];
        double repl = config[i_repl];

        if(
               0 <= cont and cont < _foundry.continuators   .size()
           and 0 <= crat and crat < _foundry.crossover_rates.size()
           and 0 <= cros and cros < _foundry.crossovers     .size()
           and 0 <= mrat and mrat < _foundry.mutation_rates .size()
           and 0 <= muta and muta < _foundry.mutations      .size()
           and 0 <= sele and sele < _foundry.selectors      .size()
           and 0 <= pops and pops < _foundry.pop_sizes      .size()
           and 0 <= repl and repl < _foundry.replacements   .size()
        ) {
            _foundry.select(config);

            // FIXME should pop_size belong to this eval and moved out from the foundry?
            // Reset pop
            eoPop<SUB> pop;
            pop.append( _foundry.pop_size(), _subpb_init);
            _subpb_eval(pop,pop);

            // Actually perform a search
            _foundry(pop);

            sol.fitness( pop.best_element().fitness() );

        } else {
            eo::log << eo::warnings << "WARNING: encoded algo is out of bounds" << std::endl;
            sol.fitness( _penalization ); // penalization
        }
    }

protected:
    eoInit<SUB>& _subpb_init;
    eoPopEvalFunc<SUB>& _subpb_eval;
    eoAlgoFoundryFastGA<SUB>& _foundry;
    const typename EOT::Fitness _penalization;
};

/** Helper function to instanciate an eoEvalFoundryFastGA without having to indicate the template for the sub-problem encoding.
 *
 * The template is deduced from the constructor's parameters.
 * Not sure it's more concise than a classical instanciation…
 */
template<class EOT, class SUB>
eoEvalFoundryFastGA<EOT,SUB>&
    make_eoEvalFoundryFastGA(
        eoInit<SUB>& subpb_init,
        eoPopEvalFunc<SUB>& subpb_eval,
        eoAlgoFoundryFastGA<SUB>& foundry,
        const typename SUB::Fitness penalization    )
{
    return *(new eoEvalFoundryFastGA<EOT,SUB>(subpb_init, subpb_eval, foundry, penalization));
}

#endif // _eoEvalFoundryFastGA_H_

