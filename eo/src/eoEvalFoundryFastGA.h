
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
            const typename SUB::Fitness penalization,
            const bool normalized = false
        ) :
            _subpb_init(subpb_init),
            _subpb_eval(subpb_eval),
            _foundry(foundry),
            _penalization(penalization),
            _normalized(normalized),
            i_crat(foundry.crossover_rates.index()),
            i_crsl(foundry.crossover_selectors.index()),
            i_cros(foundry.crossovers.index()),
            i_afcr(foundry.aftercross_selectors.index()),
            i_mrat(foundry.mutation_rates.index()),
            i_musl(foundry.mutation_selectors.index()),
            i_muta(foundry.mutations.index()),
            i_repl(foundry.replacements.index()),
            i_cont(foundry.continuators.index()),
            i_pops(foundry.pop_sizes.index())
    { }

protected:
    const size_t i_crat;
    const size_t i_crsl;
    const size_t i_cros;
    const size_t i_afcr;
    const size_t i_mrat;
    const size_t i_musl;
    const size_t i_muta;
    const size_t i_repl;
    const size_t i_cont;
    const size_t i_pops;

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
        size_t crat;
        size_t crsl;
        size_t cros;
        size_t afcr;
        size_t mrat;
        size_t musl;
        size_t muta;
        size_t repl;
        size_t cont;
        size_t pops;

        if(_normalized) {
            crat = static_cast<size_t>(std::ceil( sol[i_crat] * _foundry.crossover_rates.size() ));
            crsl = static_cast<size_t>(std::ceil( sol[i_crsl] * _foundry.crossover_selectors.size() ));
            cros = static_cast<size_t>(std::ceil( sol[i_cros] * _foundry.crossovers.size() ));
            afcr = static_cast<size_t>(std::ceil( sol[i_afcr] * _foundry.aftercross_selectors.size() ));
            mrat = static_cast<size_t>(std::ceil( sol[i_mrat] * _foundry.mutation_rates.size() ));
            musl = static_cast<size_t>(std::ceil( sol[i_musl] * _foundry.mutation_selectors.size() ));
            muta = static_cast<size_t>(std::ceil( sol[i_muta] * _foundry.mutations.size() ));
            repl = static_cast<size_t>(std::ceil( sol[i_repl] * _foundry.replacements.size() ));
            cont = static_cast<size_t>(std::ceil( sol[i_cont] * _foundry.continuators.size() ));
            pops = static_cast<size_t>(std::ceil( sol[i_pops] * _foundry.pop_sizes.size() ));

        } else {
            crat = static_cast<size_t>(std::ceil( sol[i_crat] ));
            crsl = static_cast<size_t>(std::ceil( sol[i_crsl] ));
            cros = static_cast<size_t>(std::ceil( sol[i_cros] ));
            afcr = static_cast<size_t>(std::ceil( sol[i_afcr] ));
            mrat = static_cast<size_t>(std::ceil( sol[i_mrat] ));
            musl = static_cast<size_t>(std::ceil( sol[i_musl] ));
            muta = static_cast<size_t>(std::ceil( sol[i_muta] ));
            repl = static_cast<size_t>(std::ceil( sol[i_repl] ));
            cont = static_cast<size_t>(std::ceil( sol[i_cont] ));
            pops = static_cast<size_t>(std::ceil( sol[i_pops] ));
        }
        return {crat, crsl, cros, afcr, mrat, musl, muta, repl, cont, pops};
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
        double crat = config[i_crat];
        double crsl = config[i_crsl];
        double cros = config[i_cros];
        double afcr = config[i_afcr];
        double mrat = config[i_mrat];
        double musl = config[i_musl];
        double muta = config[i_muta];
        double repl = config[i_repl];
        double cont = config[i_cont];
        double pops = config[i_pops];

        if(
                0 <= crat and crat < _foundry.crossover_rates.size()
            and 0 <= crsl and crsl < _foundry.crossover_selectors.size()
            and 0 <= cros and cros < _foundry.crossovers.size()
            and 0 <= afcr and afcr < _foundry.aftercross_selectors.size()
            and 0 <= mrat and mrat < _foundry.mutation_rates.size()
            and 0 <= musl and musl < _foundry.mutation_selectors.size()
            and 0 <= muta and muta < _foundry.mutations.size()
            and 0 <= repl and repl < _foundry.replacements.size()
            and 0 <= cont and cont < _foundry.continuators.size()
            and 0 <= pops and pops < _foundry.pop_sizes.size()
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
    const bool _normalized;
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
        const typename SUB::Fitness penalization,
        const bool normalized = false )
{
    return *(new eoEvalFoundryFastGA<EOT,SUB>(subpb_init, subpb_eval, foundry, penalization, normalized));
}

#endif // _eoEvalFoundryFastGA_H_

